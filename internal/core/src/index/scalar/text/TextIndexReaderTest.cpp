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

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "index/growing/TantivyGrowingTextIndex.h"
#include "index/scalar/text/TextIndexReader.h"
#include "index/test_utils/ArtifactTestUtils.h"
#include "index/test_utils/ScalarTestData.h"

namespace milvus::index::test {

class GrowingTextValidityBitmapTest : public ::testing::Test {
 protected:
    static std::unique_ptr<TantivyGrowingTextIndex>
    Owner() {
        return std::make_unique<TantivyGrowingTextIndex>(
            "text-validity",
            "milvus_tokenizer",
            R"({"tokenizer":"standard"})",
            DataType::VARCHAR,
            1000000);
    }

    static const SharedValidityBitmap::Snapshot&
    Validity(const GrowingIndexSnapshotPin& pin) {
        return dynamic_cast<const TextIndexReader&>(pin.Reader()).Validity();
    }

    static void
    AppendNulls(TantivyGrowingTextIndex& owner, size_t begin, size_t count) {
        const std::vector<std::string_view> values(count);
        const auto valid = std::make_unique<bool[]>(count);
        owner.Append(static_cast<int64_t>(begin),
                     {count, values.data(), valid.get()});
    }

    static void
    ExpectSnapshot(const GrowingIndexSnapshotPin& pin,
                   const std::vector<uint8_t>& valid,
                   const std::vector<size_t>& matches = {}) {
        ASSERT_TRUE(static_cast<bool>(pin));
        EXPECT_EQ(pin.CoveredRowEnd(), valid.size());
        EXPECT_EQ(pin.Reader().Count(), valid.size());
        const auto& reader = dynamic_cast<const TextIndexReader&>(pin.Reader());
        const auto is_null = reader.IsNull();
        const auto is_not_null = reader.IsNotNull();
        const auto actual_matches = reader.MatchQuery("needle", 0);
        ASSERT_EQ(is_null.size(), valid.size());
        ASSERT_EQ(is_not_null.size(), valid.size());
        ASSERT_EQ(actual_matches.size(), valid.size());
        TargetBitmap expected_matches(valid.size());
        for (const auto row : matches) {
            expected_matches.set(row);
        }
        for (size_t row = 0; row < valid.size(); ++row) {
            EXPECT_EQ(static_cast<bool>(is_null[row]), valid[row] == 0) << row;
            EXPECT_EQ(static_cast<bool>(is_not_null[row]), valid[row] != 0)
                << row;
            EXPECT_EQ(actual_matches[row], expected_matches[row]) << row;
        }
    }
};

namespace {

TEST_F(GrowingTextValidityBitmapTest,
       RepeatedPublicationsShareBufferAndFreezePartialWord) {
    auto owner = Owner();
    const std::array<std::string_view, 4> values{"needle", "", "needle", ""};
    const std::array<bool, 4> first_valid{true, true, false, false};
    owner->Append(0, {values.size(), values.data(), first_valid.data()});
    owner->Flush();
    const auto first = owner->PinSnapshot();
    ExpectSnapshot(first, {1, 1, 0, 0}, {0});
    const std::array<bool, 4> second_valid{false, true, true, false};
    owner->Append(4, {values.size(), values.data(), second_valid.data()});
    owner->Flush();
    const auto second = owner->PinSnapshot();
    ASSERT_NE(Validity(first).FullWords().data(), nullptr);
    EXPECT_EQ(Validity(first).FullWords().data(),
              Validity(second).FullWords().data());
    EXPECT_EQ(Validity(first).BufferBytes(), 128 * 1024);
    owner->Append(8, {values.size(), values.data(), nullptr});
    owner->Flush();
    const auto third = owner->PinSnapshot();
    ExpectSnapshot(first, {1, 1, 0, 0}, {0});
    ExpectSnapshot(second, {1, 1, 0, 0, 0, 1, 1, 0}, {0, 6});
    ExpectSnapshot(third, {1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1}, {0, 6, 8, 10});
}

TEST_F(GrowingTextValidityBitmapTest,
       SharesWordsAcrossBoundaryAndWriterLifetime) {
    constexpr size_t kWordBits = 64;
    auto owner = Owner();
    AppendNulls(*owner, 0, kWordBits - 1);
    owner->Flush();
    const auto partial = owner->PinSnapshot();
    const std::vector<uint8_t> partial_valid(kWordBits - 1, 0);
    const std::array<std::string_view, 3> values{"needle", "", ""};
    const std::array<bool, 3> crossing_valid{true, false, true};
    owner->Append(kWordBits - 1,
                  {values.size(), values.data(), crossing_valid.data()});
    owner->Flush();
    const auto full = owner->PinSnapshot();
    auto full_valid = partial_valid;
    full_valid.insert(full_valid.end(), {1, 0, 1});
    EXPECT_EQ(Validity(partial).FullWords().data(),
              Validity(full).FullWords().data());
    const std::array<std::string_view, 3> tail_values{"", "needle", ""};
    const std::array<bool, 3> tail_valid{false, true, false};
    owner->Append(kWordBits + 2,
                  {tail_values.size(), tail_values.data(), tail_valid.data()});
    owner->Flush();
    const auto tail = owner->PinSnapshot();
    auto tail_expected = full_valid;
    tail_expected.insert(tail_expected.end(), {0, 1, 0});
    EXPECT_EQ(Validity(full).FullWords().data(),
              Validity(tail).FullWords().data());
    owner.reset();
    ExpectSnapshot(partial, partial_valid);
    ExpectSnapshot(full, full_valid, {kWordBits - 1});
    ExpectSnapshot(tail, tail_expected, {kWordBits - 1, kWordBits + 3});
}

TEST_F(GrowingTextValidityBitmapTest, OneAppendSpansSeveralValidityWords) {
    constexpr size_t kWordBits = 64;
    auto owner = Owner();
    AppendNulls(*owner, 0, 2 * kWordBits + 7);
    owner->Flush();
    const auto first = owner->PinSnapshot();
    AppendNulls(*owner, 2 * kWordBits + 7, kWordBits - 7);
    owner->Flush();
    const auto second = owner->PinSnapshot();
    EXPECT_EQ(Validity(first).FullWords().data(),
              Validity(second).FullWords().data());
    ExpectSnapshot(first, std::vector<uint8_t>(2 * kWordBits + 7, 0));
    ExpectSnapshot(second, std::vector<uint8_t>(3 * kWordBits, 0));
}

TEST_F(GrowingTextValidityBitmapTest, AllValidPrefixAllocatesOnlyAtFirstNull) {
    auto owner = Owner();
    const std::array<std::string_view, 2> values{"needle", ""};
    owner->Append(0, {values.size(), values.data(), nullptr});
    owner->Flush();
    const auto all_valid = owner->PinSnapshot();
    EXPECT_TRUE(Validity(all_valid).AllValid());
    EXPECT_EQ(Validity(all_valid).BufferBytes(), 0);
    AppendNulls(*owner, 2, 1);
    owner->Flush();
    const auto nullable = owner->PinSnapshot();
    EXPECT_FALSE(Validity(nullable).AllValid());
    ExpectSnapshot(all_valid, {1, 1}, {0});
    ExpectSnapshot(nullable, {1, 1, 0}, {0});
}

TEST_F(GrowingTextValidityBitmapTest,
       FailedAppendRollbackPreservesAcceptedNullBits) {
    auto owner = Owner();
    const std::array<std::string_view, 2> accepted{"needle", ""};
    const std::array<bool, 2> valid{true, false};
    owner->Append(0, {accepted.size(), accepted.data(), valid.data()});
    owner->Flush();
    const auto first = owner->PinSnapshot();
    owner->Append(2, {accepted.size(), accepted.data(), valid.data()});
    // The first two writes mutate the engine, then the real Rust UTF-8 check
    // rejects the final value. Rollback must replay accepted rows 2 and 3 only.
    const std::string invalid_utf8(1, static_cast<char>(0xff));
    const std::array<std::string_view, 3> rejected{"", "needle", invalid_utf8};
    const std::array<bool, 3> rejected_valid{false, true, true};
    EXPECT_THROW(
        owner->Append(
            4, {rejected.size(), rejected.data(), rejected_valid.data()}),
        SegcoreError);
    ExpectSnapshot(owner->PinSnapshot(), {1, 0}, {0});
    owner->Flush();
    const auto recovered = owner->PinSnapshot();
    ExpectSnapshot(recovered, {1, 0, 1, 0}, {0, 2});
    const std::array<std::string_view, 3> retry{"needle", "", "needle"};
    const std::array<bool, 3> retry_valid{true, false, true};
    owner->Append(4, {retry.size(), retry.data(), retry_valid.data()});
    owner->Flush();
    const auto final = owner->PinSnapshot();
    ExpectSnapshot(final, {1, 0, 1, 0, 1, 0, 1}, {0, 2, 4, 6});
    ExpectSnapshot(first, {1, 0}, {0});
    ExpectSnapshot(recovered, {1, 0, 1, 0}, {0, 2});
}

TEST(TextSealedValidityTest, LoadsNullableAndAllValidSnapshots) {
    for (const bool nullable : {false, true}) {
        for (const auto name : {"TextVarcharV7Heap", "TextVarcharV7Mmap"}) {
            const auto& backend =
                ScalarReaderBackends().Get<std::string_view>(name);
            ScalarTestData<std::string_view> data({"needle", "", "needle"});
            if (nullable) {
                data.validity.reset(1);
            } else {
                data.validity_present = false;
            }
            const ScalarTestInput<std::string_view> input(data);
            auto artifact = backend.Build(input.View(), {.row_count = 3});
            auto reader = backend.Open(std::move(artifact), {.row_count = 3});
            const auto* text =
                dynamic_cast<const TextIndexReader*>(reader.get());
            ASSERT_NE(text, nullptr);
            EXPECT_EQ(text->Validity().AllValid(), !nullable);
            const auto is_null = text->IsNull();
            const auto is_not_null = text->IsNotNull();
            const auto matches = text->MatchQuery("needle", 1);
            ASSERT_EQ(is_null.size(), 3);
            for (size_t row = 0; row < 3; ++row) {
                EXPECT_EQ(is_null[row], nullable && row == 1) << row;
                EXPECT_EQ(is_not_null[row], !nullable || row != 1) << row;
                EXPECT_EQ(matches[row], row != 1) << row;
            }
            if (!nullable) {
                EXPECT_EQ(text->Validity().BufferBytes(), 0);
            }
        }
    }
}

}  // namespace
}  // namespace milvus::index::test
