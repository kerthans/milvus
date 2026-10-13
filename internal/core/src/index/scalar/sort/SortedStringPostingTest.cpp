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

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "index/contracts/query/INullReader.h"
#include "index/contracts/query/IScalarPredicateReader.h"
#include "index/contracts/query/IScalarValueReader.h"
#include "index/scalar/sort/SortedIndexReader.h"
#include "index/test_utils/ArtifactTestUtils.h"
#include "index/test_utils/ScalarTestData.h"

namespace milvus::index::test {
namespace {

void
ExpectPostingRoundTrip(std::vector<std::string> input_values,
                       bool nested = false) {
    ScalarTestData<std::string_view> data(std::move(input_values));
    const auto count = data.values.size();
    ASSERT_GT(count, 4);
    data.validity.reset(1);
    data.validity.reset(count - 1);
    data.batch_sizes = {2, 0, count / 2, count - count / 2 - 2};
    const ScalarTestInput<std::string_view> input(data);
    const std::vector<std::string_view> keys{
        data.values.front(), data.values[count / 2], "absent"};
    const std::string_view lower = "key-1";
    const std::string_view upper = "key-3";
    const std::vector<const char*> backends =
        nested ? std::vector<const char*>{"SortedVarcharNested",
                                          "SortedVarcharNestedMmap"}
               : std::vector<const char*>{"SortedVarchar", "SortedVarcharMmap"};
    for (const auto* name : backends) {
        SCOPED_TRACE(name);
        const auto& backend =
            ScalarReaderBackends().Get<std::string_view>(name);
        auto artifact = backend.Build(input.View(), {.row_count = count});
        for (const bool packed : {false, true}) {
            SCOPED_TRACE(packed);
            auto reader = packed ? OpenV3(backend, SerializeV3(*artifact))
                                 : OpenV1V2(backend, SerializeV1V2(*artifact));
            ASSERT_NE(reader, nullptr);
            EXPECT_EQ(reader->Count(), count);
            const auto* predicate =
                dynamic_cast<const IScalarPredicateReader<std::string_view>*>(
                    reader.get());
            const auto* values =
                dynamic_cast<const IScalarValueReader<std::string_view>*>(
                    reader.get());
            const auto* nulls = dynamic_cast<const INullReader*>(reader.get());
            ASSERT_NE(predicate, nullptr);
            ASSERT_NE(values, nullptr);
            ASSERT_NE(nulls, nullptr);
            const auto hits = predicate->In(keys.size(), keys.data());
            const auto misses = predicate->NotIn(keys.size(), keys.data());
            const auto range = predicate->Range(lower, true, upper, false);
            const auto is_null = nulls->IsNull();
            ASSERT_EQ(hits.size(), count);
            ASSERT_EQ(misses.size(), count);
            ASSERT_EQ(range.size(), count);
            ASSERT_EQ(is_null.size(), count);
            for (size_t row = 0; row < count; ++row) {
                const std::string_view value = data.values[row];
                const bool valid = nested || data.validity[row];
                const bool match =
                    std::find(keys.begin(), keys.end(), value) != keys.end();
                EXPECT_EQ(hits[row], valid && match) << row;
                EXPECT_EQ(misses[row], valid && !match) << row;
                EXPECT_EQ(range[row], valid && value >= lower && value < upper)
                    << row;
                EXPECT_EQ(is_null[row], !valid) << row;
                const auto lookup = values->Lookup(row);
                EXPECT_EQ(lookup.has_value(), valid) << row;
                if (valid) {
                    ASSERT_TRUE(lookup.has_value());
                    EXPECT_EQ(*lookup, value) << row;
                }
            }
        }
    }
}

TEST(SortedStringPostingTest, HighCardinalityBuildLoadQuery) {
    std::vector<std::string> values;
    // Each posting has one through four row IDs, including the inline limit.
    for (size_t key = 0; key < 64; ++key) {
        for (size_t repeat = 0; repeat <= key % 4; ++repeat) {
            values.push_back("key-" + std::to_string(key));
        }
    }
    ExpectPostingRoundTrip(std::move(values));
}

TEST(SortedStringPostingTest, LowCardinalityBuildLoadQuery) {
    std::vector<std::string> values;
    for (size_t row = 0; row < 288; ++row) {
        values.push_back("key-" + std::to_string(row % 3));
    }
    ExpectPostingRoundTrip(std::move(values));
}

TEST(SortedStringPostingTest, RepeatedLongKeysBuildLoadQuery) {
    const std::string key = "key-" + std::string(128, 'x');
    std::vector<std::string> values(24, key);
    values[6] += "suffix";
    values[12] = "key-" + std::string(128, 'y');
    ExpectPostingRoundTrip(std::move(values));
}

TEST(SortedStringPostingTest, NestedRepeatedLongKeysBuildLoadQuery) {
    const std::string key = "key-" + std::string(128, 'x');
    std::vector<std::string> values(24, key);
    values[6] += "suffix";
    values[12] = "key-" + std::string(128, 'y');
    ExpectPostingRoundTrip(std::move(values), true);
}

TEST(SortedStringPostingTest, ArrayRepeatedLongKeysBuildLoadQuery) {
    const std::string key = "key-" + std::string(128, 'x');
    const std::string other = key + "suffix";
    ScalarTestData<ArrayView> data({
        OwnedArrayValue::FromStrings(DataType::VARCHAR, {key, key}),
        OwnedArrayValue::FromStrings(DataType::VARCHAR, {}),
        OwnedArrayValue::FromStrings(DataType::VARCHAR, {other, key}),
        OwnedArrayValue::FromStrings(DataType::VARCHAR, {key}),
        OwnedArrayValue::FromStrings(DataType::VARCHAR, {other}),
    });
    data.validity.reset(3);
    data.batch_sizes = {1, 0, 2, 2};
    const ScalarTestInput<ArrayView> input(data);
    for (const auto* name : {"SortedVarcharArray", "SortedVarcharArrayMmap"}) {
        SCOPED_TRACE(name);
        const auto& backend = ScalarReaderBackends().Get<ArrayView>(name);
        auto artifact = backend.Build(input.View(), {.row_count = 5});
        for (const bool packed : {false, true}) {
            SCOPED_TRACE(packed);
            auto reader = packed ? OpenV3(backend, SerializeV3(*artifact))
                                 : OpenV1V2(backend, SerializeV1V2(*artifact));
            const auto* predicate =
                dynamic_cast<const IScalarPredicateReader<std::string_view>*>(
                    reader.get());
            const auto* nulls = dynamic_cast<const INullReader*>(reader.get());
            ASSERT_NE(predicate, nullptr);
            ASSERT_NE(nulls, nullptr);
            const std::string_view key_view = key;
            const std::string_view other_view = other;
            ExpectHits(predicate->In(1, &key_view), 5, {0, 2});
            ExpectHits(predicate->In(1, &other_view), 5, {2, 4});
            ExpectHits(nulls->IsNull(), 5, {3});
            EXPECT_EQ(reader->Count(), 5);
        }
    }
}

TEST(SortedStringPostingTest, HeapMemoryCountsInlinePostingStorageOnce) {
    std::vector<std::string> values{"a", "b"};
    std::vector<SortedStringPostingList> postings{{0, 1, 2, 3}, {4}};
    const auto expected = values.capacity() * sizeof(std::string) +
                          postings.capacity() * sizeof(SortedStringPostingList);
    auto layout =
        SortedStringLayout::FromHeap(std::move(values), std::move(postings), 5);
    EXPECT_EQ(layout->MemoryUsage(), expected);
    EXPECT_EQ(layout->Posting(0).size, 4);
    EXPECT_EQ(layout->Posting(0).At(3), 3);
}

TEST(SortedStringPostingTest, HeapMemoryIncludesSpilledPostingStorage) {
    std::vector<std::string> values{"a"};
    std::vector<SortedStringPostingList> postings{{0, 1, 2, 3, 4, 5, 6, 7}};
    const auto expected =
        values.capacity() * sizeof(std::string) +
        postings.capacity() * sizeof(SortedStringPostingList) +
        postings.front().capacity() * sizeof(uint32_t);
    auto layout =
        SortedStringLayout::FromHeap(std::move(values), std::move(postings), 8);
    EXPECT_EQ(layout->MemoryUsage(), expected);
    EXPECT_EQ(layout->Posting(0).At(7), 7);
}

}  // namespace
}  // namespace milvus::index::test
