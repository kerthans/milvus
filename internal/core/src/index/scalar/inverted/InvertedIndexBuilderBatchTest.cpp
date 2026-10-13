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
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "index/contracts/query/INullReader.h"
#include "index/contracts/query/IScalarPredicateReader.h"
#include "index/test_utils/ArtifactTestUtils.h"
#include "index/test_utils/ScalarTestData.h"

namespace milvus::index::test {
namespace {

TEST(InvertedIndexBuilderBatchTest, ScalarNullOffsetsSpanManySmallBatches) {
    std::vector<int64_t> values(320);
    for (size_t row = 0; row < values.size(); ++row) {
        values[row] = static_cast<int64_t>(row % 7);
    }
    ScalarTestData<int64_t> data(std::move(values));
    data.batch_sizes.assign(64, 5);
    data.batch_sizes.insert(data.batch_sizes.begin() + 17, 0);
    for (size_t row = 0; row < data.values.size(); ++row) {
        if (row % 3 == 1) {
            data.validity.reset(row);
        }
    }
    const auto& backend = ScalarReaderBackends().Get<int64_t>("InvertedInt64");
    const ScalarTestInput<int64_t> input(data);
    auto artifact = backend.Build(input.View());
    auto reader = OpenV3(backend, SerializeV3(*artifact), {.row_count = 320});
    const auto* nulls = dynamic_cast<const INullReader*>(reader.get());
    const auto* predicate =
        dynamic_cast<const IScalarPredicateReader<int64_t>*>(reader.get());
    ASSERT_NE(nulls, nullptr);
    ASSERT_NE(predicate, nullptr);
    const auto is_null = nulls->IsNull();
    const auto is_not_null = nulls->IsNotNull();
    const int64_t key = 2;
    const auto hits = predicate->In(1, &key);
    ASSERT_EQ(is_null.size(), data.values.size());
    ASSERT_EQ(hits.size(), data.values.size());
    for (size_t row = 0; row < data.values.size(); ++row) {
        EXPECT_EQ(is_null[row], !data.validity[row]) << row;
        EXPECT_EQ(is_not_null[row], data.validity[row]) << row;
        EXPECT_EQ(hits[row], data.validity[row] && data.values[row] == key)
            << row;
    }
}

TEST(InvertedIndexBuilderBatchTest, ArrayNullAndEmptyRowsSpanManySmallBatches) {
    const std::array<int64_t, 2> nonempty{1, 3};
    const std::array<int64_t, 0> empty{};
    std::vector<OwnedArrayValue> values;
    for (size_t row = 0; row < 320; ++row) {
        values.push_back(row % 3 == 0 ? OwnedArrayValue::FromFixed<int64_t>(
                                            DataType::INT64, empty)
                                      : OwnedArrayValue::FromFixed<int64_t>(
                                            DataType::INT64, nonempty));
    }
    ScalarTestData<ArrayView> data(std::move(values));
    data.batch_sizes.assign(64, 5);
    for (size_t row = 0; row < data.values.size(); ++row) {
        if (row % 3 == 1) {
            data.validity.reset(row);
        }
    }
    const auto& backend =
        ScalarReaderBackends().Get<ArrayView>("InvertedInt64Array");
    const ScalarTestInput<ArrayView> input(data);
    auto artifact = backend.Build(input.View());
    auto reader = OpenV3(backend, SerializeV3(*artifact), {.row_count = 320});
    const auto* nulls = dynamic_cast<const INullReader*>(reader.get());
    const auto* predicate =
        dynamic_cast<const IScalarPredicateReader<int64_t>*>(reader.get());
    ASSERT_NE(nulls, nullptr);
    ASSERT_NE(predicate, nullptr);
    const auto is_null = nulls->IsNull();
    const auto is_not_null = nulls->IsNotNull();
    const int64_t key = 1;
    const auto hits = predicate->In(1, &key);
    ASSERT_EQ(is_null.size(), data.values.size());
    ASSERT_EQ(hits.size(), data.values.size());
    for (size_t row = 0; row < data.values.size(); ++row) {
        EXPECT_EQ(is_null[row], row % 3 == 1) << row;
        EXPECT_EQ(is_not_null[row], row % 3 != 1) << row;
        EXPECT_EQ(hits[row], row % 3 == 2) << row;
    }
}

TEST(InvertedIndexBuilderBatchTest, NestedBatchesKeepElementCoordinates) {
    ScalarTestData<int64_t> data({0, 1, 2, 2, 3, 2, 4, 2, 5});
    data.validity_present = false;
    data.domain = Domain::Element;
    data.batch_sizes = {1, 0, 2, 1, 2, 1, 2};
    const auto& backend =
        ScalarReaderBackends().Get<int64_t>("InvertedInt64Nested");
    for (const bool validity_present : {false, true}) {
        SCOPED_TRACE(validity_present);
        data.validity_present = validity_present;
        const ScalarTestInput<int64_t> input(data);
        auto reader = backend.Create(input.View());
        ASSERT_EQ(reader->CoordDomain(), Domain::Element);
        const auto* nulls = dynamic_cast<const INullReader*>(reader.get());
        const auto* predicate =
            dynamic_cast<const IScalarPredicateReader<int64_t>*>(reader.get());
        ASSERT_NE(nulls, nullptr);
        ASSERT_NE(predicate, nullptr);
        EXPECT_EQ(nulls->IsNull().count(), 0);
        EXPECT_EQ(nulls->IsNotNull().count(), data.values.size());
        const int64_t key = 2;
        const auto hits = predicate->In(1, &key);
        ASSERT_EQ(hits.size(), data.values.size());
        for (size_t row = 0; row < data.values.size(); ++row) {
            EXPECT_EQ(hits[row], data.values[row] == key) << row;
        }
    }
}

}  // namespace
}  // namespace milvus::index::test
