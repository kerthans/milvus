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

#include "index/Meta.h"
#include "index/contracts/query/IJsonIndexReader.h"
#include "index/contracts/query/INullReader.h"
#include "index/contracts/query/IScalarPredicateReader.h"
#include "index/test_utils/ArtifactTestUtils.h"
#include "index/test_utils/ScalarTestData.h"

namespace milvus::index::test {
namespace {

TEST(JsonFlatIndexBuilderTest, FieldNullOffsetsSpanManySmallBatches) {
    std::vector<std::string> values;
    for (size_t row = 0; row < 320; ++row) {
        values.push_back(row % 4 == 2 ? R"({"other":7})"
                                      : R"({"profile":{"age":7}})");
    }
    ScalarTestData<std::string_view> data(std::move(values));
    data.batch_sizes.assign(64, 5);
    data.batch_sizes.insert(data.batch_sizes.begin() + 11, 0);
    for (size_t row = 0; row < data.values.size(); ++row) {
        if (row % 4 == 1) {
            data.validity.reset(row);
        }
    }
    const auto& backend =
        ScalarReaderBackends().Get<std::string_view>("JsonFlatV7");
    const BackendCaseMetadata metadata{.row_count = data.values.size(),
                                       .values = {{JSON_PATH, "/profile"}}};
    const ScalarTestInput<std::string_view> input(data);
    auto artifact = backend.Build(input.View(), metadata);
    auto reader = OpenV3(backend, SerializeV3(*artifact), metadata);
    const auto* field_nulls = dynamic_cast<const INullReader*>(reader.get());
    const auto* json = dynamic_cast<const IJsonIndexReader*>(reader.get());
    ASSERT_NE(field_nulls, nullptr);
    ASSERT_NE(json, nullptr);
    auto resolved =
        json->Resolve("/profile/age", JsonCastType::FromString("DOUBLE"));
    ASSERT_TRUE(resolved);
    const auto* predicate =
        dynamic_cast<const IScalarPredicateReader<double>*>(resolved.get());
    ASSERT_NE(predicate, nullptr);
    const auto is_null = field_nulls->IsNull();
    const auto is_not_null = field_nulls->IsNotNull();
    const double key = 7;
    const auto hits = predicate->In(1, &key);
    ASSERT_EQ(is_null.size(), data.values.size());
    ASSERT_EQ(hits.size(), data.values.size());
    for (size_t row = 0; row < data.values.size(); ++row) {
        EXPECT_EQ(is_null[row], row % 4 == 1) << row;
        EXPECT_EQ(is_not_null[row], row % 4 != 1) << row;
        EXPECT_EQ(hits[row], row % 4 == 0 || row % 4 == 3) << row;
    }
}

}  // namespace
}  // namespace milvus::index::test
