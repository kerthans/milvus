// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <arrow/filesystem/localfs.h>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "folly/ScopeGuard.h"
#include "folly/coro/BlockingWait.h"
#include "common/Consts.h"
#include "common/OpContext.h"
#include "index/Families.h"
#include "index/Meta.h"
#include "index/contracts/query/INullReader.h"
#include "index/contracts/query/ITextMatchReader.h"
#include "segcore/storagev1translator/TextMatchIndexTranslator.h"
#include "storage/LoadAdmissionController.h"
#include "storage/MemFileManagerImpl.h"
#include "storage/test_utils/TextPublicationTestUtils.h"
#include "test_utils/AsyncLoadTestUtils.h"

namespace milvus::segcore::storagev1translator {
namespace {

class TextMetadataFileSystem final : public arrow::fs::SubTreeFileSystem {
 public:
    TextMetadataFileSystem(std::shared_ptr<arrow::fs::FileSystem> fs,
                           std::vector<uint8_t> bytes)
        : SubTreeFileSystem("", std::move(fs)), bytes_(std::move(bytes)) {
    }

    arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>>
    OpenInputFile(const std::string&) override {
        return Open();
    }

    arrow::Future<std::shared_ptr<arrow::io::RandomAccessFile>>
    OpenInputFileAsync(const std::string&) override {
        return arrow::Future<
            std::shared_ptr<arrow::io::RandomAccessFile>>::MakeFinished(Open());
    }

    size_t opens{0};
    folly::CancellationSource* cancel_on_open{nullptr};
    bool fail_open{false};
    std::vector<std::shared_ptr<milvus::test::ControlledDirectReadFile>> files;

 private:
    arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>>
    Open() {
        ++opens;
        if (cancel_on_open) {
            cancel_on_open->requestCancellation();
        }
        if (fail_open) {
            return arrow::Status::IOError(
                "injected text metadata open failure");
        }
        auto file =
            std::make_shared<milvus::test::ControlledDirectReadFile>(bytes_);
        files.push_back(file);
        return std::shared_ptr<arrow::io::RandomAccessFile>(std::move(file));
    }

    const std::vector<uint8_t> bytes_;
};

class TextMatchIndexMetadataContextTest : public ::testing::Test {
 protected:
    void
    SetUp() override {
        fixture_ =
            std::make_unique<storage::text_test::TextPublicationFixture>();
        const auto stats = fixture_->Publish(true);
        config_ = fixture_->LoadConfig(stats, false);
        info_ = {false, 3, 101, "{}", stats.MemSize(), 3, "disable", ""};
        storage::MemFileManagerImpl manager(fixture_->context);
        const auto paths =
            config_.at(index::INDEX_FILES).get<std::vector<std::string>>();
        auto input = manager.OpenInputStream(paths.front(), false);
        std::vector<uint8_t> bytes(input->Size());
        input->ReadAt(bytes.data(), 0, bytes.size());
        fs_ = std::make_shared<TextMetadataFileSystem>(fixture_->context.fs,
                                                       std::move(bytes));
        fixture_->context.fs = fs_;
    }

    std::unique_ptr<TextMatchIndexTranslator>
    Construct(OpContext* operation = nullptr) const {
        return std::make_unique<TextMatchIndexTranslator>(
            info_, fixture_->context, config_, operation);
    }

    void
    ExpectAdmissionPriority(OpContext* operation, bool low) {
        auto& budget = storage::LoadAdmissionController::GetInstance();
        const auto saved_slots = budget.CapacitySlots();
        budget.SetCapacitySlots(1);
        auto restore =
            folly::makeGuard([&] { budget.SetCapacitySlots(saved_slots); });
        auto occupied = folly::coro::blockingWait(
            budget.AcquireAsync({0, 1}, storage::LoadAdmissionPriority::High));
        auto loading = std::async(std::launch::async,
                                  [&] { return Construct(operation); });
        auto join = folly::makeGuard([&] {
            occupied.Release();
            if (loading.valid()) {
                loading.wait();
            }
        });
        EXPECT_TRUE(WaitForQueuedAdmission(budget))
            << "text metadata must enter the shared admission queue";
        // HIGH may bypass queued LOW, but never an earlier queued HIGH.
        const bool high_bypasses =
            budget.TryAcquire({0, 0}, storage::LoadAdmissionPriority::High);
        EXPECT_EQ(high_bypasses, low);
        if (high_bypasses) {
            budget.Release({0, 0});
        }
        occupied.Release();
        const auto translator = loading.get();
        ASSERT_NE(translator, nullptr);
        EXPECT_EQ(translator->Family(), index::families::kText);
    }

    static bool
    WaitForQueuedAdmission(storage::LoadAdmissionController& budget) {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            if (!budget.TryAcquire({0, 0},
                                   storage::LoadAdmissionPriority::Low)) {
                return true;
            }
            budget.Release({0, 0});
            std::this_thread::yield();
        }
        return false;
    }

    std::unique_ptr<storage::text_test::TextPublicationFixture> fixture_;
    TextMatchIndexLoadInfo info_{};
    Config config_;
    std::shared_ptr<TextMetadataFileSystem> fs_;
};

TEST_F(TextMatchIndexMetadataContextTest,
       ConfiguredPriorityWithoutRuntimeControlsMetadataAdmission) {
    OpContext operation;
    ASSERT_FALSE(operation.runtime_load_priority.has_value());
    for (const int32_t configured : {0, 1}) {
        config_[LOAD_PRIORITY] = configured;
        for (const bool async : {false, true}) {
            fixture_->context.use_async_load = async;
            ExpectAdmissionPriority(&operation, configured != 0);
            EXPECT_FALSE(operation.runtime_load_priority.has_value());
        }
    }
}

TEST_F(TextMatchIndexMetadataContextTest,
       RuntimePriorityOverridesConfiguredMetadataPriority) {
    OpContext operation;
    for (const int32_t configured : {0, 1}) {
        config_[LOAD_PRIORITY] = configured;
        operation.runtime_load_priority = configured == 0 ? 1 : 0;
        for (const bool async : {false, true}) {
            fixture_->context.use_async_load = async;
            ExpectAdmissionPriority(&operation, configured == 0);
            EXPECT_EQ(operation.runtime_load_priority, configured == 0 ? 1 : 0);
        }
    }
}

TEST_F(TextMatchIndexMetadataContextTest,
       NullContextUsesConfiguredMetadataPriority) {
    for (const int32_t configured : {0, 1}) {
        config_[LOAD_PRIORITY] = configured;
        for (const bool async : {false, true}) {
            fixture_->context.use_async_load = async;
            ExpectAdmissionPriority(nullptr, configured != 0);
        }
    }
}

TEST_F(TextMatchIndexMetadataContextTest, AlreadyCancelledDoesNotOpenMetadata) {
    folly::CancellationSource cancelled;
    cancelled.requestCancellation();
    OpContext operation(cancelled.getToken());
    config_[LOAD_PRIORITY] = 1;
    for (const bool async : {false, true}) {
        fixture_->context.use_async_load = async;
        try {
            static_cast<void>(Construct(&operation));
            FAIL() << "cancelled construction must fail";
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), ErrorCode::FollyCancel);
        }
        EXPECT_EQ(fs_->opens, 0);
        EXPECT_TRUE(fs_->files.empty());
    }
}

TEST_F(TextMatchIndexMetadataContextTest,
       CancellationAfterOpenStopsMetadataReads) {
    for (const bool async : {false, true}) {
        for (const bool fail : {false, true}) {
            fixture_->context.use_async_load = async;
            folly::CancellationSource cancelled;
            OpContext operation(cancelled.getToken());
            fs_->cancel_on_open = &cancelled;
            fs_->fail_open = fail;
            fs_->opens = 0;
            fs_->files.clear();
            try {
                static_cast<void>(Construct(&operation));
                FAIL() << "cancelled metadata open must fail";
            } catch (const SegcoreError& error) {
                EXPECT_EQ(error.get_error_code(), ErrorCode::FollyCancel);
            }
            EXPECT_EQ(fs_->opens, 1);
            for (const auto& file : fs_->files) {
                EXPECT_TRUE(file->DirectReadCalls().empty());
            }
            fs_->cancel_on_open = nullptr;
        }
    }
}

TEST_F(TextMatchIndexMetadataContextTest,
       CancellationWhileMetadataWaitsForAdmission) {
    auto& budget = storage::LoadAdmissionController::GetInstance();
    const auto saved_slots = budget.CapacitySlots();
    budget.SetCapacitySlots(1);
    auto restore =
        folly::makeGuard([&] { budget.SetCapacitySlots(saved_slots); });
    config_[LOAD_PRIORITY] = 1;
    for (const bool async : {false, true}) {
        fixture_->context.use_async_load = async;
        folly::CancellationSource cancelled;
        OpContext operation(cancelled.getToken());
        auto occupied = folly::coro::blockingWait(
            budget.AcquireAsync({0, 1}, storage::LoadAdmissionPriority::High));
        auto loading = std::async(std::launch::async,
                                  [&] { return Construct(&operation); });
        auto join = folly::makeGuard([&] {
            occupied.Release();
            if (loading.valid()) {
                loading.wait();
            }
        });
        EXPECT_TRUE(WaitForQueuedAdmission(budget));
        cancelled.requestCancellation();
        EXPECT_EQ(loading.wait_for(std::chrono::seconds(5)),
                  std::future_status::ready);
        occupied.Release();
        try {
            static_cast<void>(loading.get());
            FAIL() << "cancelled metadata admission must fail";
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), ErrorCode::FollyCancel);
        }
    }
}

TEST_F(TextMatchIndexMetadataContextTest,
       ConstructionDoesNotRetainCallerContext) {
    fixture_->context.use_async_load = true;
    config_[LOAD_PRIORITY] = 1;
    std::unique_ptr<TextMatchIndexTranslator> translator;
    {
        OpContext caller;
        translator = Construct(&caller);
    }
    const auto cells = translator->get_cells(nullptr, {0});
    ASSERT_EQ(cells.size(), 1);
    ASSERT_NE(cells[0].second, nullptr);
    EXPECT_EQ(cells[0].second->Count(), 3);
    const auto* text =
        dynamic_cast<const index::ITextMatchReader*>(cells[0].second.get());
    const auto* nulls =
        dynamic_cast<const index::INullReader*>(cells[0].second.get());
    ASSERT_NE(text, nullptr);
    ASSERT_NE(nulls, nullptr);
    const auto hits = text->MatchQuery("alpha", 1);
    ASSERT_EQ(hits.size(), 3);
    EXPECT_TRUE(hits[0]);
    EXPECT_EQ(hits.count(), 1);
    const auto is_null = nulls->IsNull();
    EXPECT_EQ(is_null.count(), 1);
    EXPECT_TRUE(is_null[1]);
}

}  // namespace
}  // namespace milvus::segcore::storagev1translator
