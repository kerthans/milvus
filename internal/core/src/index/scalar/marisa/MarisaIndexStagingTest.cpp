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

#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <utility>

#include "index/Families.h"
#include "index/Meta.h"
#include "index/contracts/Registry.h"
#include "index/contracts/query/IScalarPredicateReader.h"
#include "index/test_utils/ArtifactTestUtils.h"
#include "index/test_utils/ScalarTestData.h"
#include "storage/artifact/LocalDirectory.h"

namespace milvus::index::test {
namespace {

class ScopedTmpDir final {
 public:
    explicit ScopedTmpDir(const std::string& path) {
        if (const auto* previous = std::getenv("TMPDIR")) {
            previous_ = previous;
        }
        if (::setenv("TMPDIR", path.c_str(), 1) != 0) {
            throw std::runtime_error("failed to set test TMPDIR");
        }
    }

    ~ScopedTmpDir() {
        if (previous_) {
            ::setenv("TMPDIR", previous_->c_str(), 1);
        } else {
            ::unsetenv("TMPDIR");
        }
    }

 private:
    std::optional<std::string> previous_;
};

storage::ArtifactPtr
BuildInDirectory(const std::string& local_dir) {
    ScalarTestData<std::string_view> data({"alpha", "beta", "beta"});
    data.validity_present = false;
    const ScalarTestInput<std::string_view> input(data);
    auto builder =
        BuilderRegistry<ScalarBuildInput<std::string_view>>::Instance().Create(
            families::kMarisa,
            {{"field_type", static_cast<int>(DataType::VARCHAR)},
             {"value_type", static_cast<int>(DataType::VARCHAR)},
             {"local_dir", local_dir}});
    if (!builder) {
        throw std::logic_error("MARISA builder is not registered");
    }
    return std::move(*builder).Build(input.View());
}

// Observe the real serialization descriptor before delegating its bytes. The
// named entry must already be gone while its open descriptor remains usable.
class ObserveTrieWriter final : public storage::IndexEntryWriter {
 public:
    ObserveTrieWriter(TestArtifactData& output,
                      std::string local_dir,
                      std::string temp_dir,
                      bool fail,
                      bool recreate_name = false)
        : inner_(output),
          local_dir_(std::move(local_dir)),
          temp_dir_(std::move(temp_dir)),
          fail_(fail),
          recreate_name_(recreate_name) {
    }

    void
    WriteEntry(const std::string& name,
               const void* data,
               size_t size) override {
        inner_.WriteEntry(name, data, size);
    }

    void
    WriteEntry(const std::string& name, int fd, size_t size) override {
        if (name == MARISA_TRIE_INDEX) {
            ++trie_writes;
            trie_fd = fd;
            struct stat status {};
            ASSERT_EQ(::fstat(fd, &status), 0);
            EXPECT_GT(status.st_size, 0);
            EXPECT_EQ(status.st_nlink, 0);
            const auto flags = ::fcntl(fd, F_GETFD);
            ASSERT_NE(flags, -1);
            EXPECT_NE(flags & FD_CLOEXEC, 0);
            const auto path = std::filesystem::read_symlink("/proc/self/fd/" +
                                                            std::to_string(fd))
                                  .string();
            EXPECT_EQ(path.find(local_dir_ + "/marisa_trie_"), 0);
            EXPECT_EQ(path.find(temp_dir_ + "/"), std::string::npos);
            EXPECT_TRUE(std::filesystem::is_empty(local_dir_));
            EXPECT_TRUE(std::filesystem::is_empty(temp_dir_));
            if (recreate_name_) {
                constexpr std::string_view deleted_suffix = " (deleted)";
                ASSERT_TRUE(path.ends_with(deleted_suffix));
                replacement_path =
                    path.substr(0, path.size() - deleted_suffix.size());
                std::ofstream replacement(replacement_path, std::ios::binary);
                ASSERT_TRUE(replacement.is_open());
                replacement << "replacement";
                replacement.close();
                ASSERT_FALSE(replacement.fail());
            }
            if (fail_) {
                ThrowInfo(FileWriteFailed,
                          "injected MARISA entry sink failure");
            }
        }
        inner_.WriteEntry(name, fd, size);
    }

    void
    Finish() override {
        for (const auto& [key, value] : meta_json_.items()) {
            inner_.PutMeta(key, value);
        }
        inner_.Finish();
    }

    size_t
    GetTotalBytesWritten() const override {
        return inner_.GetTotalBytesWritten();
    }

    size_t trie_writes{0};
    int trie_fd{-1};
    std::string replacement_path;

 private:
    TestArtifactWriter inner_;
    std::string local_dir_;
    std::string temp_dir_;
    bool fail_;
    bool recreate_name_;
};

TEST(MarisaIndexStagingTest, PackedTrieUsesLocalDirAndUnlinksBeforeStreaming) {
    auto root = storage::LocalDirectory::CreateOwned(
        std::filesystem::temp_directory_path().string(),
        "marisa-staging-success-XXXXXX",
        "marisa staging test");
    const auto local_dir = root->Path() + "/configured";
    const auto temp_dir = root->Path() + "/system-temp";
    std::filesystem::create_directory(temp_dir);
    ScopedTmpDir temp(temp_dir);
    // Leave local_dir absent so creation of the configured parent is covered.
    auto artifact = BuildInDirectory(local_dir);
    TestArtifactData persisted;
    ObserveTrieWriter sink(persisted, local_dir, temp_dir, false);
    artifact->Serialize(sink);
    sink.Finish();
    ASSERT_EQ(sink.trie_writes, 1);
    errno = 0;
    const auto fd_status = ::fcntl(sink.trie_fd, F_GETFD);
    const auto fd_error = errno;
    EXPECT_EQ(fd_status, -1);
    EXPECT_EQ(fd_error, EBADF);
    EXPECT_TRUE(std::filesystem::is_empty(local_dir));
    EXPECT_TRUE(std::filesystem::is_empty(temp_dir));
    const auto& backend =
        ScalarReaderBackends().Get<std::string_view>("MarisaVarcharNonNull");
    auto reader = OpenV3(backend, persisted, {.row_count = 3});
    const auto* predicate =
        dynamic_cast<const IScalarPredicateReader<std::string_view>*>(
            reader.get());
    ASSERT_NE(predicate, nullptr);
    const std::string_view key = "beta";
    const auto hits = predicate->In(1, &key);
    EXPECT_EQ(hits.count(), 2);
    EXPECT_TRUE(hits[1]);
    EXPECT_TRUE(hits[2]);
}

TEST(MarisaIndexStagingTest, PackedTrieClosesAndCleansAfterSinkFailure) {
    auto root = storage::LocalDirectory::CreateOwned(
        std::filesystem::temp_directory_path().string(),
        "marisa-staging-failure-XXXXXX",
        "marisa staging failure test");
    const auto local_dir = root->Path() + "/configured";
    const auto temp_dir = root->Path() + "/system-temp";
    std::filesystem::create_directory(temp_dir);
    ScopedTmpDir temp(temp_dir);
    auto artifact = BuildInDirectory(local_dir);
    TestArtifactData persisted;
    ObserveTrieWriter sink(persisted, local_dir, temp_dir, true);
    ExpectSegcoreError(ErrorCode::FileWriteFailed,
                       [&] { artifact->Serialize(sink); });
    ASSERT_EQ(sink.trie_writes, 1);
    errno = 0;
    const auto fd_status = ::fcntl(sink.trie_fd, F_GETFD);
    const auto fd_error = errno;
    EXPECT_EQ(fd_status, -1);
    EXPECT_EQ(fd_error, EBADF);
    EXPECT_TRUE(std::filesystem::is_empty(local_dir));
    EXPECT_TRUE(std::filesystem::is_empty(temp_dir));
}

TEST(MarisaIndexStagingTest, PackedTriePreservesRecreatedName) {
    for (const bool fail : {false, true}) {
        SCOPED_TRACE(fail);
        auto root = storage::LocalDirectory::CreateOwned(
            std::filesystem::temp_directory_path().string(),
            "marisa-staging-reused-name-XXXXXX",
            "marisa staging reused name test");
        const auto local_dir = root->Path() + "/configured";
        const auto temp_dir = root->Path() + "/system-temp";
        std::filesystem::create_directory(temp_dir);
        ScopedTmpDir temp(temp_dir);
        auto artifact = BuildInDirectory(local_dir);
        TestArtifactData persisted;
        ObserveTrieWriter sink(persisted, local_dir, temp_dir, fail, true);
        if (fail) {
            ExpectSegcoreError(ErrorCode::FileWriteFailed,
                               [&] { artifact->Serialize(sink); });
        } else {
            artifact->Serialize(sink);
            sink.Finish();
        }
        ASSERT_EQ(sink.trie_writes, 1);
        errno = 0;
        const auto fd_status = ::fcntl(sink.trie_fd, F_GETFD);
        const auto fd_error = errno;
        EXPECT_EQ(fd_status, -1);
        EXPECT_EQ(fd_error, EBADF);
        ASSERT_FALSE(sink.replacement_path.empty());
        ASSERT_TRUE(std::filesystem::is_regular_file(sink.replacement_path));
        std::ifstream replacement(sink.replacement_path, std::ios::binary);
        std::string contents;
        std::getline(replacement, contents);
        EXPECT_EQ(contents, "replacement");
        EXPECT_TRUE(std::filesystem::is_empty(temp_dir));
    }
}

TEST(MarisaIndexStagingTest, PackedTrieCreateFailureReportsDirectoryAndErrno) {
    auto root = storage::LocalDirectory::CreateOwned(
        std::filesystem::temp_directory_path().string(),
        "marisa-staging-create-error-XXXXXX",
        "marisa staging create failure test");
    const auto file_path = root->Path() + "/regular-file";
    {
        std::ofstream file(file_path);
        ASSERT_TRUE(file.is_open());
    }
    // A regular-file parent fails create_directories; procfs is an existing
    // directory that rejects mkostemp even when this test runs as root.
    for (const auto& local_dir :
         {file_path + "/child", std::string("/proc/self")}) {
        SCOPED_TRACE(local_dir);
        auto artifact = BuildInDirectory(local_dir);
        TestArtifactData persisted;
        TestArtifactWriter sink(persisted);
        try {
            artifact->Serialize(sink);
            FAIL() << "expected a staging creation error";
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), ErrorCode::FileCreateFailed);
            const std::string message = error.what();
            EXPECT_NE(message.find(local_dir), std::string::npos);
            EXPECT_NE(message.find("errno="), std::string::npos);
        }
        EXPECT_TRUE(persisted.entries.empty());
    }
}

TEST(MarisaIndexStagingTest, LegacySerializationKeepsSystemTemporaryDirectory) {
    auto root = storage::LocalDirectory::CreateOwned(
        std::filesystem::temp_directory_path().string(),
        "marisa-staging-legacy-XXXXXX",
        "marisa legacy staging test");
    const auto local_dir = root->Path() + "/configured";
    const auto temp_dir = root->Path() + "/system-temp";
    std::filesystem::create_directory(temp_dir);
    ScopedTmpDir temp(temp_dir);
    auto artifact = BuildInDirectory(local_dir);
    const auto persisted = SerializeV1V2(*artifact);
    EXPECT_FALSE(std::filesystem::exists(local_dir));
    EXPECT_TRUE(std::filesystem::is_empty(temp_dir));
    EXPECT_GT(persisted.entries.at(MARISA_TRIE_INDEX).size(), 0);
}

}  // namespace
}  // namespace milvus::index::test
