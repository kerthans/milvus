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

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <unistd.h>
#include <variant>
#include <vector>

#include "index/contracts/query/INullReader.h"
#include "index/contracts/query/IScalarPredicateReader.h"
#include "index/contracts/query/IScalarValueReader.h"
#include "index/scalar/sort/IndexStructure.h"
#include "index/scalar/sort/SortedIndexFormat.h"
#include "index/scalar/sort/SortedIndexLoader.h"
#include "index/test_utils/ArtifactTestUtils.h"
#include "index/test_utils/LoaderTestAccess.h"
#include "index/test_utils/ScalarTestData.h"
#include "storage/artifact/LocalDirectory.h"

namespace milvus::index::test {
namespace {

template <typename T>
ScalarTestData<T>
AuxData() {
    ScalarTestData<T> data = [] {
        if constexpr (std::is_same_v<T, std::string_view>) {
            return ScalarTestData<T>({"c", "a", "b", "a", "unused"});
        } else {
            return ScalarTestData<T>({3, 1, 2, 1, 0});
        }
    }();
    data.validity.reset(4);
    return data;
}

template <typename T>
TestArtifactData
AuxArtifact(const ReaderBackend& backend) {
    const auto data = AuxData<T>();
    const ScalarTestInput<T> input(data);
    auto artifact = backend.Build(input.View(), {.row_count = 5});
    return SerializeV3(*artifact);
}

template <typename T>
void
ExpectAuxRoundTrip(std::string_view base_name) {
    const auto data = AuxData<T>();
    for (const auto* suffix : {"", "Mmap"}) {
        const auto& backend =
            ScalarReaderBackends().Get<T>(std::string(base_name) + suffix);
        for (const bool persisted : {false, true}) {
            SCOPED_TRACE(backend.Name() +
                         ":persisted=" + std::to_string(persisted));
            auto artifact = AuxArtifact<T>(backend);
            if (!persisted) {
                artifact.entries.erase(std::string(sort_format::kIdxToOffsets));
                if constexpr (!std::is_same_v<T, std::string_view>) {
                    artifact.entries.erase(
                        std::string(sort_format::kValidBitset));
                }
            }
            for (const bool async : {false, true}) {
                SCOPED_TRACE(async);
                auto reader =
                    OpenV3(backend, artifact, {.row_count = 5}, async);
                const auto* predicate =
                    dynamic_cast<const IScalarPredicateReader<T>*>(
                        reader.get());
                const auto* values =
                    dynamic_cast<const IScalarValueReader<T>*>(reader.get());
                const auto* nulls =
                    dynamic_cast<const INullReader*>(reader.get());
                ASSERT_NE(predicate, nullptr);
                ASSERT_NE(values, nullptr);
                ASSERT_NE(nulls, nullptr);
                const T key = data.values[1];
                const auto in = predicate->In(1, &key);
                const auto not_in = predicate->NotIn(1, &key);
                const auto is_null = nulls->IsNull();
                ASSERT_EQ(in.size(), 5);
                ASSERT_EQ(not_in.size(), 5);
                ASSERT_EQ(is_null.size(), 5);
                for (size_t row = 0; row < 5; ++row) {
                    const bool valid = data.validity[row];
                    EXPECT_EQ(in[row], valid && data.values[row] == key) << row;
                    EXPECT_EQ(not_in[row], valid && data.values[row] != key)
                        << row;
                    EXPECT_EQ(is_null[row], !valid) << row;
                    const auto lookup = values->Lookup(row);
                    ASSERT_EQ(lookup.has_value(), valid) << row;
                    if (valid) {
                        EXPECT_EQ(*lookup, data.values[row]) << row;
                    }
                }
            }
        }
    }
}

size_t
FirstPostingRow(const std::vector<uint8_t>& bytes) {
    auto read = [&](size_t at) {
        uint32_t value;
        std::memcpy(&value, bytes.data() + at, sizeof(value));
        return value;
    };
    const auto unique = read(0);
    const auto strings = sizeof(uint32_t) + unique * sizeof(uint32_t);
    const auto last = strings + read(unique * sizeof(uint32_t));
    const auto posting_offsets = last + sizeof(uint32_t) + read(last);
    return posting_offsets + unique * sizeof(uint32_t) + read(posting_offsets) +
           sizeof(uint32_t);
}

TEST(SortedPersistedAuxTest, NumericHeapAndMmapRoundTrip) {
    ExpectAuxRoundTrip<int64_t>("SortedInt64");
}

TEST(SortedPersistedAuxTest, StringHeapAndMmapRoundTrip) {
    ExpectAuxRoundTrip<std::string_view>("SortedVarchar");
}

TEST(SortedPersistedAuxTest, NumericRebuildRejectsOutOfRangeCoordinates) {
    for (const auto* name : {"SortedInt64", "SortedInt64Mmap"}) {
        const auto& backend = ScalarReaderBackends().Get<int64_t>(name);
        for (const int32_t invalid : {-1, 5}) {
            SCOPED_TRACE(invalid);
            auto artifact = AuxArtifact<int64_t>(backend);
            artifact.entries.erase(std::string(sort_format::kIdxToOffsets));
            artifact.entries.erase(std::string(sort_format::kValidBitset));
            auto& data =
                artifact.entries.at(std::string(sort_format::kIndexData));
            IndexStructure<int64_t> entry;
            ASSERT_GE(data.size(), sizeof(entry));
            std::memcpy(&entry, data.data(), sizeof(entry));
            entry.idx_ = invalid;
            std::memcpy(data.data(), &entry, sizeof(entry));
            ExpectSegcoreError(ErrorCode::DataFormatBroken, [&] {
                static_cast<void>(OpenV3(backend, artifact, {.row_count = 5}));
            });
        }
    }
}

TEST(SortedPersistedAuxTest, StringRebuildRejectsOutOfRangeCoordinates) {
    for (const auto* name : {"SortedVarchar", "SortedVarcharMmap"}) {
        const auto& backend =
            ScalarReaderBackends().Get<std::string_view>(name);
        auto artifact = AuxArtifact<std::string_view>(backend);
        artifact.entries.erase(std::string(sort_format::kIdxToOffsets));
        auto& data = artifact.entries.at(std::string(sort_format::kIndexData));
        const uint32_t invalid_row = 5;
        std::memcpy(data.data() + FirstPostingRow(data),
                    &invalid_row,
                    sizeof(invalid_row));
        ExpectSegcoreError(ErrorCode::DataFormatBroken, [&] {
            static_cast<void>(OpenV3(backend, artifact, {.row_count = 5}));
        });
    }
}

TEST(SortedPersistedAuxTest, PersistedNumericSemanticContentsAreTrusted) {
    for (const auto* name : {"SortedInt64", "SortedInt64Mmap"}) {
        const auto& backend = ScalarReaderBackends().Get<int64_t>(name);
        auto artifact = AuxArtifact<int64_t>(backend);
        auto& data = artifact.entries.at(std::string(sort_format::kIndexData));
        IndexStructure<int64_t> entry;
        std::memcpy(&entry, data.data(), sizeof(entry));
        entry.a_ = 999;
        entry.idx_ = 99;
        std::memcpy(data.data(), &entry, sizeof(entry));
        auto& offsets =
            artifact.entries.at(std::string(sort_format::kIdxToOffsets));
        const int32_t invalid = std::numeric_limits<int32_t>::max();
        std::memcpy(offsets.data(), &invalid, sizeof(invalid));
        // The wire sizes are valid. Loading trusts the persisted tables and
        // does not inspect order, row IDs, or reverse-offset contents.
        auto reader = OpenV3(backend, artifact, {.row_count = 5});
        EXPECT_EQ(reader->Count(), 5);
    }
}

TEST(SortedPersistedAuxTest, PersistedStringSemanticContentsAreTrusted) {
    for (const auto* name : {"SortedVarchar", "SortedVarcharMmap"}) {
        const auto& backend =
            ScalarReaderBackends().Get<std::string_view>(name);
        auto artifact = AuxArtifact<std::string_view>(backend);
        auto& data = artifact.entries.at(std::string(sort_format::kIndexData));
        const uint32_t invalid_row = 99;
        std::memcpy(data.data() + FirstPostingRow(data),
                    &invalid_row,
                    sizeof(invalid_row));
        auto& offsets =
            artifact.entries.at(std::string(sort_format::kIdxToOffsets));
        const int32_t invalid = std::numeric_limits<int32_t>::max();
        std::memcpy(offsets.data(), &invalid, sizeof(invalid));
        // Do not query this intentionally inconsistent artifact: only loading
        // and metadata are part of the persisted-aux trust contract.
        auto reader = OpenV3(backend, artifact, {.row_count = 5});
        EXPECT_EQ(reader->Count(), 5);
    }
}

template <typename T>
void
ExpectUntouchedMmapFiles(const char* backend_name) {
    const auto& backend = ScalarReaderBackends().Get<T>(backend_name);
    const auto artifact = AuxArtifact<T>(backend);
    auto input = std::make_shared<storage::RemoteInputStream>(
        std::make_shared<arrow::io::BufferReader>(
            MakePackedArtifactBuffer(artifact)));
    auto source = storage::IndexEntryReader::Open(input, input->Size());
    auto staging = storage::LocalDirectory::CreateOwned(
        std::filesystem::temp_directory_path().string(),
        "sorted-aux-unread-XXXXXX",
        "sorted lazy auxiliary test");
    storage::LoadOptions options;
    options.enable_mmap = true;
    options.mmap_dir_path = staging->Path();
    options.params = backend.LoadParams({.row_count = 5});
    auto plan = LoaderTestAccess::Plan<SortedIndexLoader>(
        source->Directory(), source->IndexMeta(), options);
    const auto files = storage::CollectIndexFileTargets(plan.entries);
    for (const auto& file : files) {
        file->Prepare(storage::io::Priority::HIGH);
    }
    for (const auto& entry : plan.entries) {
        const auto& bytes = artifact.entries.at(entry.name);
        if (const auto* memory =
                std::get_if<storage::MemoryEntryTarget>(&entry.target)) {
            std::memcpy(memory->data, bytes.data(), bytes.size());
        } else {
            const auto& file = std::get<storage::FileEntryTarget>(entry.target);
            file.staging->WriteAt(file.offset, bytes.data(), bytes.size());
        }
    }
    for (const auto& file : files) {
        file->Finish();
    }
    const auto& offsets = std::get<storage::FileEntryTarget>(
        plan.At(sort_format::kIdxToOffsets).target);
    ASSERT_EQ(::truncate(offsets.staging->path.c_str(), 0), 0);
    if constexpr (!std::is_same_v<T, std::string_view>) {
        const auto& data = std::get<storage::FileEntryTarget>(
            plan.At(sort_format::kIndexData).target);
        ASSERT_EQ(::truncate(data.staging->path.c_str(), 0), 0);
    }
    // Linux permits these file mappings, but any payload dereference raises
    // SIGBUS. The child must finish and destroy the reader without touching
    // numeric data/offsets or string offsets; string header pages remain valid.
    auto reader = LoaderTestAccess::Finish<SortedIndexLoader>(plan, options);
    ASSERT_NE(reader, nullptr);
    EXPECT_EQ(reader->Count(), 5);
}

TEST(SortedPersistedAuxDeathTest, MmapNumericFinalizationDoesNotTouchPayloads) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_EXIT(
        {
            ExpectUntouchedMmapFiles<int64_t>("SortedInt64Mmap");
            ::_exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST(SortedPersistedAuxDeathTest, MmapStringFinalizationDoesNotTouchOffsets) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_EXIT(
        {
            ExpectUntouchedMmapFiles<std::string_view>("SortedVarcharMmap");
            ::_exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

}  // namespace
}  // namespace milvus::index::test
