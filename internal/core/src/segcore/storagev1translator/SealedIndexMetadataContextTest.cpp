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
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <vector>

#include "folly/ScopeGuard.h"
#include "folly/coro/BlockingWait.h"
#include "common/Consts.h"
#include "common/OpContext.h"
#include "index/Families.h"
#include "index/IndexTypeAdapter.h"
#include "index/contracts/Registry.h"
#include "index/contracts/build/ScalarBuildInput.h"
#include "segcore/Types.h"
#include "segcore/storagev1translator/SealedIndexTranslator.h"
#include "storage/LoadAdmissionController.h"
#include "storage/MemFileManagerImpl.h"
#include "storage/Util.h"
#include "storage/artifact/LocalDirectory.h"
#include "test_utils/AsyncLoadTestUtils.h"

namespace milvus::segcore::storagev1translator {
namespace {

class MetadataFileSystem final : public arrow::fs::SubTreeFileSystem {
 public:
    MetadataFileSystem(std::shared_ptr<arrow::fs::FileSystem> fs,
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
            return arrow::Status::IOError("injected metadata open failure");
        }
        auto file =
            std::make_shared<milvus::test::ControlledDirectReadFile>(bytes_);
        files.push_back(file);
        return std::shared_ptr<arrow::io::RandomAccessFile>(std::move(file));
    }

    const std::vector<uint8_t> bytes_;
};

class SealedIndexMetadataContextTest : public ::testing::Test {
 protected:
    void
    SetUp() override {
        root_ = storage::LocalDirectory::CreateOwned(
            std::filesystem::temp_directory_path().string(),
            "metadata-context-XXXXXX",
            "metadata context test");
        storage::StorageConfig config;
        config.storage_type = "local";
        config.root_path = root_->Path();
        storage::FieldDataMeta field{1, 2, 3, 101};
        field.field_schema.set_data_type(proto::schema::DataType::Int32);
        context_ =
            storage::FileManagerContext(field,
                                        storage::IndexMeta{3, 101, 1000, 10000},
                                        storage::CreateChunkManager(config),
                                        storage::InitArrowFileSystem(config));
        context_.set_for_loading_index(true);
        const std::vector<int32_t> values{30, 10, 20, 10};
        const index::ScalarBuildBatch<int32_t> batch{values, {}};
        auto builder =
            index::BuilderRegistry<index::ScalarBuildInput<int32_t>>::Instance()
                .Create(index::families::kSort,
                        {{"field_type", DataType::INT32},
                         {"value_type", DataType::INT32},
                         {"nested", false},
                         {"nullable", false}});
        auto artifact = std::move(*builder).Build({std::span(&batch, 1)});
        storage::MemFileManagerImpl manager(context_);
        const auto name =
            index::PackedScalarIndexFileName(index::ScalarIndexType::STLSORT);
        auto writer = manager.CreateIndexEntryWriterUnified(name);
        artifact->Serialize(*writer);
        writer->Finish();
        const auto path = manager.GetRemoteIndexObjectPrefix() + "/" + name;
        auto bytes = milvus::test::ReadPackedIndexBytes(context_, {path});
        fs_ =
            std::make_shared<MetadataFileSystem>(context_.fs, std::move(bytes));
        context_.fs = fs_;
        info_.collection_id = 1;
        info_.partition_id = 2;
        info_.segment_id = 3;
        info_.field_id = 101;
        info_.field_type = DataType::INT32;
        info_.element_type = DataType::NONE;
        info_.index_id = 1000;
        info_.index_build_id = 1000;
        info_.index_version = 10000;
        info_.index_engine_version = 3;
        info_.index_params = {{index::INDEX_TYPE, index::ASCENDING_SORT},
                              {index::SCALAR_INDEX_ENGINE_VERSION, "3"}};
        info_.index_files = {path};
        info_.index_size = 4096;
        info_.num_rows = values.size();
        info_.dim = 0;
        info_.warmup_policy = "disable";
    }

    std::unique_ptr<SealedIndexTranslator>
    Construct(OpContext* operation = nullptr) const {
        Config config(info_.index_params);
        if (configured_priority_) {
            config[LOAD_PRIORITY] = *configured_priority_;
        }
        return std::make_unique<SealedIndexTranslator>(&info_,
                                                       tracer::TraceContext{},
                                                       context_,
                                                       std::move(config),
                                                       operation);
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
            << "metadata must enter the shared admission queue";
        // A zero-resource HIGH probe can bypass a queued LOW request. It
        // cannot bypass an earlier HIGH request in the same FIFO queue.
        const bool high_bypasses =
            budget.TryAcquire({0, 0}, storage::LoadAdmissionPriority::High);
        EXPECT_EQ(high_bypasses, low);
        if (high_bypasses) {
            budget.Release({0, 0});
        }
        occupied.Release();
        const auto translator = loading.get();
        ASSERT_NE(translator, nullptr);
        EXPECT_EQ(translator->Family(), index::families::kSort);
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

    std::shared_ptr<storage::LocalDirectory> root_;
    storage::FileManagerContext context_;
    LoadIndexInfo info_{};
    std::optional<int32_t> configured_priority_;
    std::shared_ptr<MetadataFileSystem> fs_;
};

TEST_F(SealedIndexMetadataContextTest, LowPriorityMetadataUsesLowAdmission) {
    OpContext operation;
    operation.runtime_load_priority = 1;
    for (const bool async : {false, true}) {
        context_.use_async_load = async;
        ExpectAdmissionPriority(&operation, true);
    }
}

TEST_F(SealedIndexMetadataContextTest, NullContextKeepsHighPriority) {
    for (const bool async : {false, true}) {
        context_.use_async_load = async;
        ExpectAdmissionPriority(nullptr, false);
    }
}

TEST_F(SealedIndexMetadataContextTest,
       ConfiguredPriorityWithoutRuntimeControlsMetadataAdmission) {
    OpContext operation;
    ASSERT_FALSE(operation.runtime_load_priority.has_value());
    for (const int32_t configured : {0, 1}) {
        configured_priority_ = configured;
        for (const bool async : {false, true}) {
            context_.use_async_load = async;
            ExpectAdmissionPriority(&operation, configured != 0);
            EXPECT_FALSE(operation.runtime_load_priority.has_value());
        }
    }
}

TEST_F(SealedIndexMetadataContextTest,
       RuntimePriorityOverridesConfiguredMetadataPriority) {
    OpContext operation;
    for (const int32_t configured : {0, 1}) {
        configured_priority_ = configured;
        operation.runtime_load_priority = configured == 0 ? 1 : 0;
        for (const bool async : {false, true}) {
            context_.use_async_load = async;
            ExpectAdmissionPriority(&operation, configured == 0);
            EXPECT_EQ(operation.runtime_load_priority, configured == 0 ? 1 : 0);
        }
    }
}

TEST_F(SealedIndexMetadataContextTest,
       NullContextUsesConfiguredMetadataPriority) {
    for (const int32_t configured : {0, 1}) {
        configured_priority_ = configured;
        for (const bool async : {false, true}) {
            context_.use_async_load = async;
            ExpectAdmissionPriority(nullptr, configured != 0);
        }
    }
}

TEST_F(SealedIndexMetadataContextTest, AlreadyCancelledDoesNotOpenMetadata) {
    folly::CancellationSource cancelled;
    cancelled.requestCancellation();
    OpContext operation(cancelled.getToken());
    for (const bool async : {false, true}) {
        for (const std::string version : {"1", "3"}) {
            context_.use_async_load = async;
            info_.index_params[index::SCALAR_INDEX_ENGINE_VERSION] = version;
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
}

TEST_F(SealedIndexMetadataContextTest,
       CancellationAfterOpenStopsMetadataReads) {
    for (const bool async : {false, true}) {
        for (const bool fail : {false, true}) {
            context_.use_async_load = async;
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

TEST_F(SealedIndexMetadataContextTest,
       CancellationWhileMetadataWaitsForAdmission) {
    auto& budget = storage::LoadAdmissionController::GetInstance();
    const auto saved_slots = budget.CapacitySlots();
    budget.SetCapacitySlots(1);
    auto restore =
        folly::makeGuard([&] { budget.SetCapacitySlots(saved_slots); });
    for (const bool async : {false, true}) {
        context_.use_async_load = async;
        folly::CancellationSource cancelled;
        OpContext operation(cancelled.getToken());
        operation.runtime_load_priority = 1;
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
        // Cancellation must finish without waiting for the occupied slot.
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

TEST_F(SealedIndexMetadataContextTest, ConstructionDoesNotRetainCallerContext) {
    context_.use_async_load = true;
    std::unique_ptr<SealedIndexTranslator> translator;
    {
        OpContext caller;
        caller.runtime_load_priority = 1;
        translator = Construct(&caller);
    }
    const auto cells = translator->get_cells(nullptr, {0});
    ASSERT_EQ(cells.size(), 1);
    ASSERT_NE(cells[0].second, nullptr);
    EXPECT_EQ(cells[0].second->Count(), 4);
}

}  // namespace
}  // namespace milvus::segcore::storagev1translator
