// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file to you under
// the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <unistd.h>
#include <variant>
#include <vector>

#include "index/Meta.h"
#include "index/contracts/query/INullReader.h"
#include "index/contracts/query/IPatternMatchReader.h"
#include "index/contracts/query/IScalarPredicateReader.h"
#include "index/contracts/query/IScalarValueReader.h"
#include "index/scalar/marisa/MarisaIndexLoader.h"
#include "index/test_utils/ArtifactTestUtils.h"
#include "index/test_utils/LoaderTestAccess.h"
#include "index/test_utils/ScalarTestData.h"
#include "storage/artifact/LocalDirectory.h"

namespace milvus::index::test {
namespace {

ScalarTestData<std::string_view>
CsrData() {
    ScalarTestData<std::string_view> data(
        {"alpha", "beta", "alpha", "alphabet", "gamma", "", "unused"});
    data.validity.reset(6);
    data.batch_sizes = {2, 0, 3, 2};
    return data;
}

storage::ArtifactPtr
BuildCsrArtifact(const ReaderBackend& backend) {
    const auto data = CsrData();
    const ScalarTestInput<std::string_view> input(data);
    return backend.Build(input.View(), {.row_count = 7});
}

void
RemoveCsr(TestArtifactData& data) {
    data.entries.erase(MARISA_CSR_INDEX);
    data.entries.erase(MARISA_CSR_OFFSETS);
    data.metadata.erase("marisa_csr_format_version");
    data.metadata.erase("csr_num_keys");
}

void
ExpectCsrQueries(const IIndexReaderBase& reader) {
    const auto data = CsrData();
    const auto* predicate =
        dynamic_cast<const IScalarPredicateReader<std::string_view>*>(&reader);
    const auto* pattern = dynamic_cast<const IPatternMatchReader*>(&reader);
    const auto* values =
        dynamic_cast<const IScalarValueReader<std::string_view>*>(&reader);
    const auto* nulls = dynamic_cast<const INullReader*>(&reader);
    ASSERT_NE(predicate, nullptr);
    ASSERT_NE(pattern, nullptr);
    ASSERT_NE(values, nullptr);
    ASSERT_NE(nulls, nullptr);
    const std::string_view equal = "beta";
    const std::vector<std::string_view> terms{"alpha", "gamma", "missing"};
    const auto equality = predicate->In(1, &equal);
    const auto in = predicate->In(terms.size(), terms.data());
    const auto prefix = pattern->PatternMatch("alph", PatternOp::PrefixMatch);
    const auto is_null = nulls->IsNull();
    ASSERT_EQ(equality.size(), 7);
    ASSERT_EQ(in.size(), 7);
    ASSERT_EQ(prefix.size(), 7);
    ASSERT_EQ(is_null.size(), 7);
    for (size_t row = 0; row < data.values.size(); ++row) {
        const bool valid = data.validity[row];
        const std::string_view value = data.values[row];
        const bool match =
            std::find(terms.begin(), terms.end(), value) != terms.end();
        EXPECT_EQ(equality[row], valid && value == equal) << row;
        EXPECT_EQ(in[row], valid && match) << row;
        EXPECT_EQ(prefix[row], valid && value.starts_with("alph")) << row;
        EXPECT_EQ(is_null[row], !valid) << row;
        const auto lookup = values->Lookup(row);
        ASSERT_EQ(lookup.has_value(), valid) << row;
        if (valid) {
            EXPECT_EQ(*lookup, value) << row;
        }
    }
}

TEST(MarisaPersistedCsrTest, HeapAndMmapRoundTrip) {
    for (const auto* name : {"MarisaVarchar", "MarisaVarcharMmap"}) {
        SCOPED_TRACE(name);
        const auto& backend =
            ScalarReaderBackends().Get<std::string_view>(name);
        auto artifact = BuildCsrArtifact(backend);
        auto legacy =
            OpenV1V2(backend, SerializeV1V2(*artifact), {.row_count = 7});
        ExpectCsrQueries(*legacy);
        for (const bool persisted : {false, true}) {
            auto data = SerializeV3(*artifact);
            if (!persisted) {
                RemoveCsr(data);
            }
            for (const bool async : {false, true}) {
                SCOPED_TRACE(persisted);
                SCOPED_TRACE(async);
                auto reader = OpenV3(backend, data, {.row_count = 7}, async);
                ExpectCsrQueries(*reader);
            }
        }
    }
}

TEST(MarisaPersistedCsrTest, RebuildingCsrChecksKeyIdBounds) {
    for (const auto* name : {"MarisaVarchar", "MarisaVarcharMmap"}) {
        const auto& backend =
            ScalarReaderBackends().Get<std::string_view>(name);
        for (const int64_t invalid :
             {int64_t{-2}, std::numeric_limits<int64_t>::max()}) {
            auto data = SerializeV3(*BuildCsrArtifact(backend));
            RemoveCsr(data);
            auto& ids = data.entries.at(MARISA_STR_IDS);
            std::memcpy(ids.data(), &invalid, sizeof(invalid));
            ExpectSegcoreError(ErrorCode::DataFormatBroken, [&] {
                static_cast<void>(OpenV3(backend, data, {.row_count = 7}));
            });
        }
    }
}

TEST(MarisaPersistedCsrTest, PersistedTablesAreTrusted) {
    for (const auto* name : {"MarisaVarchar", "MarisaVarcharMmap"}) {
        const auto& backend =
            ScalarReaderBackends().Get<std::string_view>(name);
        auto data = SerializeV3(*BuildCsrArtifact(backend));
        auto& ids = data.entries.at(MARISA_STR_IDS);
        const int64_t invalid_id = std::numeric_limits<int64_t>::max();
        std::memcpy(ids.data(), &invalid_id, sizeof(invalid_id));
        auto& offsets = data.entries.at(MARISA_CSR_OFFSETS);
        ASSERT_GE(offsets.size(), sizeof(uint32_t));
        const uint32_t invalid_row = 99;
        std::memcpy(offsets.data(), &invalid_row, sizeof(invalid_row));
        // Byte counts and key metadata are intact. Loading no longer audits
        // every row ID and CSR posting. Do not query these inconsistent tables.
        auto reader = OpenV3(backend, data, {.row_count = 7});
        EXPECT_EQ(reader->Count(), 7);
    }
}

TEST(MarisaPersistedCsrTest, PackedLoadRejectsNonzeroFirstCsrEndpoint) {
    for (const auto* name : {"MarisaVarchar", "MarisaVarcharMmap"}) {
        SCOPED_TRACE(name);
        const auto& backend =
            ScalarReaderBackends().Get<std::string_view>(name);
        auto data = SerializeV3(*BuildCsrArtifact(backend));
        auto& index = data.entries.at(MARISA_CSR_INDEX);
        const uint32_t first = 1;
        std::memcpy(index.data(), &first, sizeof(first));
        for (const bool async : {false, true}) {
            SCOPED_TRACE(async);
            ExpectSegcoreError(ErrorCode::DataFormatBroken, [&] {
                static_cast<void>(
                    OpenV3(backend, data, {.row_count = 7}, async));
            });
        }
    }
}

TEST(MarisaPersistedCsrTest, PackedLoadRejectsNonzeroTerminalWithEmptyOffsets) {
    for (const auto* name : {"MarisaVarchar", "MarisaVarcharMmap"}) {
        SCOPED_TRACE(name);
        const auto& backend =
            ScalarReaderBackends().Get<std::string_view>(name);
        auto data = SerializeV3(*BuildCsrArtifact(backend));
        // Keep the original nonzero terminal entry with a zero-byte offsets
        // entry. Heap and mmap loads must reject before creating a reader.
        ASSERT_FALSE(data.entries.at(MARISA_CSR_OFFSETS).empty());
        data.entries.at(MARISA_CSR_OFFSETS).clear();
        for (const bool async : {false, true}) {
            SCOPED_TRACE(async);
            ExpectSegcoreError(ErrorCode::DataFormatBroken, [&] {
                static_cast<void>(
                    OpenV3(backend, data, {.row_count = 7}, async));
            });
        }
    }
}

TEST(MarisaPersistedCsrTest, PackedLoadRejectsMismatchedTerminalCsrEndpoint) {
    for (const auto* name : {"MarisaVarchar", "MarisaVarcharMmap"}) {
        SCOPED_TRACE(name);
        const auto& backend =
            ScalarReaderBackends().Get<std::string_view>(name);
        auto data = SerializeV3(*BuildCsrArtifact(backend));
        auto& index = data.entries.at(MARISA_CSR_INDEX);
        const uint32_t terminal =
            data.entries.at(MARISA_CSR_OFFSETS).size() / sizeof(uint32_t) + 1;
        std::memcpy(index.data() + index.size() - sizeof(terminal),
                    &terminal,
                    sizeof(terminal));
        for (const bool async : {false, true}) {
            SCOPED_TRACE(async);
            ExpectSegcoreError(ErrorCode::DataFormatBroken, [&] {
                static_cast<void>(
                    OpenV3(backend, data, {.row_count = 7}, async));
            });
        }
    }
}

TEST(MarisaPersistedCsrTest, QueryRejectsInvalidIntermediateCsrRange) {
    for (const auto* name : {"MarisaVarchar", "MarisaVarcharMmap"}) {
        SCOPED_TRACE(name);
        const auto& backend =
            ScalarReaderBackends().Get<std::string_view>(name);
        const auto original = SerializeV3(*BuildCsrArtifact(backend));
        const auto values = CsrData();
        const auto keys = original.metadata.at("csr_num_keys").get<size_t>();
        const auto& ids = original.entries.at(MARISA_STR_IDS);
        size_t key_id = 0;
        std::string_view key;
        // Read the generated IDs so the fixture does not assume MARISA's key
        // ordering. Both edited entries must be strictly between the endpoints.
        for (size_t row = 0; row < values.values.size(); ++row) {
            if (!values.validity[row]) {
                continue;
            }
            int64_t id;
            std::memcpy(&id, ids.data() + row * sizeof(id), sizeof(id));
            if (id > 0 && static_cast<size_t>(id) + 1 < keys) {
                key_id = static_cast<size_t>(id);
                key = values.values[row];
                break;
            }
        }
        ASSERT_GT(key_id, 0);
        ASSERT_LT(key_id + 1, keys);
        const uint32_t count =
            original.entries.at(MARISA_CSR_OFFSETS).size() / sizeof(uint32_t);
        for (const bool reversed : {false, true}) {
            SCOPED_TRACE(reversed);
            auto data = original;
            auto& index = data.entries.at(MARISA_CSR_INDEX);
            const uint32_t begin = reversed ? count : 0;
            const uint32_t end = reversed ? 0 : count + 1;
            std::memcpy(
                index.data() + key_id * sizeof(begin), &begin, sizeof(begin));
            std::memcpy(
                index.data() + (key_id + 1) * sizeof(end), &end, sizeof(end));
            for (const bool async : {false, true}) {
                SCOPED_TRACE(async);
                // Endpoints remain valid, so loading trusts the middle entries.
                auto reader = OpenV3(backend, data, {.row_count = 7}, async);
                const auto* predicate = dynamic_cast<
                    const IScalarPredicateReader<std::string_view>*>(
                    reader.get());
                ASSERT_NE(predicate, nullptr);
                ExpectSegcoreError(ErrorCode::DataFormatBroken, [&] {
                    static_cast<void>(predicate->In(1, &key));
                });
            }
        }
    }
}

void
ExpectUnvisitedMappings(bool unread_row_ids) {
    const auto& backend =
        ScalarReaderBackends().Get<std::string_view>("MarisaVarcharMmap");
    const auto artifact = SerializeV3(*BuildCsrArtifact(backend));
    auto input = std::make_shared<storage::RemoteInputStream>(
        std::make_shared<arrow::io::BufferReader>(
            MakePackedArtifactBuffer(artifact)));
    auto source = storage::IndexEntryReader::Open(input, input->Size());
    auto staging = storage::LocalDirectory::CreateOwned(
        std::filesystem::temp_directory_path().string(),
        "marisa-csr-unread-XXXXXX",
        "marisa lazy row-ID test");
    storage::LoadOptions options;
    options.enable_mmap = true;
    options.mmap_dir_path = staging->Path();
    options.params = backend.LoadParams({.row_count = 7});
    auto plan = LoaderTestAccess::Plan<MarisaIndexLoader>(
        source->Directory(), source->IndexMeta(), options);
    const auto files = storage::CollectIndexFileTargets(plan.entries);
    for (const auto& file : files) {
        file->Prepare(storage::io::Priority::HIGH);
    }
    for (const auto& entry : plan.entries) {
        const auto& bytes = artifact.entries.at(entry.name);
        const auto& file = std::get<storage::FileEntryTarget>(entry.target);
        file.staging->WriteAt(file.offset, bytes.data(), bytes.size());
    }
    for (const auto& file : files) {
        file->Finish();
    }
    if (unread_row_ids) {
        const auto& ids =
            std::get<storage::FileEntryTarget>(plan.At(MARISA_STR_IDS).target);
        ASSERT_EQ(::truncate(ids.staging->path.c_str(), 0), 0);
    } else {
        const auto& csr = std::get<storage::FileEntryTarget>(
            plan.At(MARISA_CSR_OFFSETS).target);
        ASSERT_GT(csr.offset, artifact.entries.at(MARISA_CSR_INDEX).size());
        ASSERT_EQ(::truncate(csr.staging->path.c_str(),
                             artifact.entries.at(MARISA_CSR_INDEX).size()),
                  0);
    }
    // mmap succeeds on Linux. Touching the truncated row-ID file or the CSR
    // posting region would SIGBUS. Keep the CSR index, including its terminal
    // entry, available for the loader's two endpoint checks and the reader's
    // constant-time pointer check. Middle entries and postings need no scan.
    auto reader = LoaderTestAccess::Finish<MarisaIndexLoader>(plan, options);
    ASSERT_NE(reader, nullptr);
    EXPECT_EQ(reader->Count(), 7);
}

TEST(MarisaPersistedCsrDeathTest, MmapFinalizationDoesNotTouchRowIds) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_EXIT(
        {
            ExpectUnvisitedMappings(true);
            ::_exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST(MarisaPersistedCsrDeathTest, MmapFinalizationDoesNotTouchCsrPostings) {
    if (static_cast<size_t>(::sysconf(_SC_PAGESIZE)) >
        storage::FileWriter::ALIGNMENT_BYTES) {
        GTEST_SKIP() << "CSR postings must start beyond the first mapped page";
    }
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_EXIT(
        {
            ExpectUnvisitedMappings(false);
            ::_exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

}  // namespace
}  // namespace milvus::index::test
