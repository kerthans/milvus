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

#include <string>
#include <utility>
#include <vector>

#include "index/contracts/query/IJsonIndexReader.h"
#include "index/contracts/query/INgramReader.h"
#include "index/contracts/query/INullReader.h"
#include "index/test_utils/ArtifactTestUtils.h"
#include "index/test_utils/ScalarTestData.h"

namespace milvus::index::test {
namespace {

TEST(NgramIndexBuilderBatchTest, ScalarNullOffsetsSpanManySmallBatches) {
    std::vector<std::string> values;
    for (size_t row = 0; row < 320; ++row) {
        values.push_back(row % 2 == 0 ? "needle value" : "other value");
    }
    ScalarTestData<std::string_view> data(std::move(values));
    data.batch_sizes.assign(64, 5);
    for (size_t row = 0; row < data.values.size(); ++row) {
        if (row % 3 == 1) {
            data.validity.reset(row);
        }
    }
    const auto& backend = ScalarReaderBackends().Get<std::string_view>(
        "NgramVarcharMin2Max4Heap");
    const ScalarTestInput<std::string_view> input(data);
    auto artifact = backend.Build(input.View());
    auto reader = OpenV3(backend, SerializeV3(*artifact), {.row_count = 320});
    const auto* nulls = dynamic_cast<const INullReader*>(reader.get());
    const auto* ngram = dynamic_cast<const INgramReader*>(reader.get());
    ASSERT_NE(nulls, nullptr);
    ASSERT_NE(ngram, nullptr);
    const auto is_null = nulls->IsNull();
    const auto is_not_null = nulls->IsNotNull();
    TargetBitmap candidates(data.values.size(), true);
    ASSERT_TRUE(ngram->CanHandle("needle", PatternOp::InnerMatch));
    ngram->Candidates("needle", PatternOp::InnerMatch, candidates);
    ASSERT_EQ(is_null.size(), data.values.size());
    for (size_t row = 0; row < data.values.size(); ++row) {
        EXPECT_EQ(is_null[row], !data.validity[row]) << row;
        EXPECT_EQ(is_not_null[row], data.validity[row]) << row;
        EXPECT_EQ(candidates[row], data.validity[row] && row % 2 == 0) << row;
    }
}

TEST(NgramIndexBuilderBatchTest, JsonTriStateOffsetsSpanManySmallBatches) {
    std::vector<OwnedJsonProjectedString> values;
    for (size_t row = 0; row < 320; ++row) {
        const auto state = row % 4 == 1   ? JsonProjectedStringState::FieldNull
                           : row % 4 == 2 ? JsonProjectedStringState::NoValue
                                          : JsonProjectedStringState::Value;
        values.push_back(
            {row % 4 == 0 ? "needle value" : "other value", state});
    }
    ScalarTestData<JsonProjectedString> data(std::move(values));
    data.validity_present = false;
    data.batch_sizes.assign(64, 5);
    data.batch_sizes.insert(data.batch_sizes.begin() + 19, 0);
    const auto& backend = ScalarReaderBackends().Get<JsonProjectedString>(
        "NgramJsonVarcharMin2Max4Heap");
    const ScalarTestInput<JsonProjectedString> input(data);
    std::vector<size_t> non_exist_offsets;
    for (size_t row = 0; row < data.values.size(); ++row) {
        if (row % 4 == 1 || row % 4 == 2) {
            non_exist_offsets.push_back(row);
        }
    }
    const BackendCaseMetadata metadata{
        .row_count = data.values.size(),
        .values = {{"non_exist_offsets", non_exist_offsets}}};
    auto artifact = backend.Build(input.View(), metadata);
    auto reader = OpenV3(backend, SerializeV3(*artifact), metadata);
    const auto* json = dynamic_cast<const IJsonIndexReader*>(reader.get());
    ASSERT_NE(json, nullptr);
    auto resolved = json->Resolve("/a", JsonCastType::FromString("VARCHAR"));
    ASSERT_TRUE(resolved);
    const auto* nulls = dynamic_cast<const INullReader*>(resolved.get());
    const auto* ngram = dynamic_cast<const INgramReader*>(resolved.get());
    ASSERT_NE(nulls, nullptr);
    ASSERT_NE(ngram, nullptr);
    const auto exists = json->Exists("/a");
    const auto is_null = nulls->IsNull();
    const auto is_not_null = nulls->IsNotNull();
    TargetBitmap candidates(data.values.size(), true);
    ASSERT_TRUE(ngram->CanHandle("needle", PatternOp::InnerMatch));
    ngram->Candidates("needle", PatternOp::InnerMatch, candidates);
    ASSERT_EQ(exists.size(), data.values.size());
    ASSERT_EQ(is_null.size(), data.values.size());
    for (size_t row = 0; row < data.values.size(); ++row) {
        EXPECT_EQ(exists[row], row % 4 == 0 || row % 4 == 3) << row;
        // NGRAM validity records field NULL; NoValue is tracked by Exists.
        EXPECT_EQ(is_null[row], row % 4 == 1) << row;
        EXPECT_EQ(is_not_null[row], row % 4 != 1) << row;
        EXPECT_EQ(candidates[row], row % 4 == 0) << row;
    }
}

}  // namespace
}  // namespace milvus::index::test
