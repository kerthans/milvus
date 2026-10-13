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

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <variant>

#include <marisa.h>

#include "index/Meta.h"
#include "index/scalar/marisa/MarisaIndexArtifact.h"
#include "index/scalar/marisa/MarisaIndexLoader.h"
#include "index/test_utils/ArtifactTestUtils.h"
#include "index/test_utils/LoaderTestAccess.h"
#include "index/test_utils/ScalarTestData.h"
#include "storage/artifact/LocalDirectory.h"

namespace milvus::index::test {
namespace {

TestArtifactData
MakeMarisaArtifact(const ReaderBackend& backend) {
    ScalarTestData<std::string_view> data({"alpha", "beta", "beta"});
    data.validity_present = false;
    const ScalarTestInput<std::string_view> input(data);
    auto artifact = backend.Build(input.View(), {.row_count = 3});
    return SerializeV3(*artifact);
}

TEST(MarisaIndexLoadErrorTest, ReadAndMmapFailuresReportFileReadFailed) {
    for (const auto* name :
         {"MarisaVarcharNonNull", "MarisaVarcharNonNullMmap"}) {
        SCOPED_TRACE(name);
        const auto& backend =
            ScalarReaderBackends().Get<std::string_view>(name);
        auto input = std::make_shared<storage::RemoteInputStream>(
            std::make_shared<arrow::io::BufferReader>(
                MakePackedArtifactBuffer(MakeMarisaArtifact(backend))));
        auto source = storage::IndexEntryReader::Open(input, input->Size());
        auto staging = storage::LocalDirectory::CreateOwned(
            std::filesystem::temp_directory_path().string(),
            "marisa-load-error-XXXXXX",
            "marisa I/O failure test");
        storage::LoadOptions options;
        options.enable_mmap = backend.MmapRequested();
        options.mmap_dir_path = staging->Path();
        options.params = backend.LoadParams({.row_count = 3});
        {
            auto plan = LoaderTestAccess::Plan<MarisaIndexLoader>(
                source->Directory(), source->IndexMeta(), options);
            const auto& trie = std::get<storage::FileEntryTarget>(
                plan.At(MARISA_TRIE_INDEX).target);
            // The staged path opens successfully, but reading a directory
            // fails with EISDIR; mapping it also fails. Both failures originate
            // inside MARISA and carry MARISA_IO_ERROR into OpenTrie.
            ASSERT_TRUE(std::filesystem::create_directory(trie.staging->path));
            ExpectSegcoreError(ErrorCode::FileReadFailed, [&] {
                static_cast<void>(
                    LoaderTestAccess::Finish<MarisaIndexLoader>(plan, options));
            });
        }
        EXPECT_TRUE(std::filesystem::is_empty(staging->Path()));
    }
}

TEST(MarisaIndexLoadErrorTest, InvalidTrieHeaderReportsDataFormatBroken) {
    for (const auto* name :
         {"MarisaVarcharNonNull", "MarisaVarcharNonNullMmap"}) {
        SCOPED_TRACE(name);
        const auto& backend =
            ScalarReaderBackends().Get<std::string_view>(name);
        auto persisted = MakeMarisaArtifact(backend);
        auto& trie = persisted.entries.at(MARISA_TRIE_INDEX);
        ASSERT_FALSE(trie.empty());
        // Keep the complete file so MARISA checks its header rather than
        // reporting an I/O error while trying to read missing header bytes.
        trie.front() ^= uint8_t{0xff};
        ExpectSegcoreError(ErrorCode::DataFormatBroken, [&] {
            static_cast<void>(OpenV3(backend, persisted, {.row_count = 3}));
        });
    }
}

TEST(MarisaIndexLoadErrorTest, MemoryAndSizeClassificationsArePreserved) {
    const marisa::Exception memory(
        __FILE__, __LINE__, MARISA_MEMORY_ERROR, "allocation failure");
    EXPECT_EQ(ClassifyMarisaError(memory, FileReadFailed, DataFormatBroken),
              ErrorCode::MemAllocateFailed);
    const marisa::Exception size(
        __FILE__, __LINE__, MARISA_SIZE_ERROR, "invalid size");
    EXPECT_EQ(ClassifyMarisaError(size, FileReadFailed, DataFormatBroken),
              ErrorCode::DataFormatBroken);
}

}  // namespace
}  // namespace milvus::index::test
