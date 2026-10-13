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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/Array.h"
#include "common/Consts.h"
#include "exec/expression/ExprIndexIntegrationTestUtils.h"
#include "exec/expression/LogicalUnaryExpr.h"
#include "exec/expression/NullExpr.h"

namespace milvus::test {
namespace {

std::shared_ptr<FieldData<Array>>
NullableArrays(DataType element_type) {
    const std::vector<std::vector<int64_t>> rows{{1}, {}, {1}, {}, {2, 3}};
    std::vector<Array> values;
    values.reserve(rows.size());
    for (const auto& row : rows) {
        proto::schema::ScalarField data;
        if (element_type == DataType::INT64) {
            auto* output = data.mutable_long_data();
            for (const auto value : row) {
                output->add_data(value);
            }
        } else {
            auto* output = data.mutable_string_data();
            for (const auto value : row) {
                output->add_data(std::to_string(value));
            }
        }
        values.emplace_back(data);
    }
    auto field = std::make_shared<FieldData<Array>>(DataType::ARRAY, true);
    const uint8_t validity = 0x16;  // NULL, [], [1], NULL, [2,3].
    field->FillFieldData(values.data(), &validity, values.size(), 0);
    return field;
}

TEST(ArrayNullIndexSelectionTest, FieldNullUsesElementIndexAndMatchesRaw) {
    constexpr size_t kRows = 5;
    for (const auto element_type : {DataType::INT64, DataType::VARCHAR}) {
        SCOPED_TRACE(static_cast<int>(element_type));
        for (const auto& family : {index::INVERTED_INDEX_TYPE,
                                   index::BITMAP_INDEX_TYPE,
                                   index::HYBRID_INDEX_TYPE}) {
            SCOPED_TRACE(family);
            auto schema = std::make_shared<Schema>();
            const auto field_id =
                schema->AddDebugArrayField("array", element_type, true);
            auto field = NullableArrays(element_type);
            expr_index::RawFieldFiles raw_files;
            auto indexed = segcore::CreateSealedSegment(schema);
            auto raw = segcore::CreateSealedSegment(schema);
            const auto raw_info = raw_files.Prepare(field_id, {field});
            indexed->LoadFieldData(raw_info);
            raw->LoadFieldData(raw_info);
            auto params = Config::object();
            if (family == index::HYBRID_INDEX_TYPE) {
                params[index::BITMAP_INDEX_CARDINALITY_LIMIT] =
                    DEFAULT_HYBRID_INDEX_BITMAP_CARDINALITY_LIMIT;
            }
            auto opened = expr_index::BuildIndex(field_id,
                                                 DataType::ARRAY,
                                                 family,
                                                 {field},
                                                 std::move(params),
                                                 element_type,
                                                 false);
            EXPECT_EQ(opened.reader->ValueType(), element_type);
            EXPECT_EQ(opened.reader->CoordDomain(), index::Domain::Row);
            OpContext untouched;
            OpContext* observed_pin = &untouched;
            expr_index::InstallIndex(*indexed,
                                     field_id,
                                     DataType::ARRAY,
                                     std::move(opened),
                                     element_type,
                                     &observed_pin);
            for (const auto op : {proto::plan::NullExpr_NullOp_IsNull,
                                  proto::plan::NullExpr_NullOp_IsNotNull}) {
                for (const bool negate : {false, true}) {
                    SCOPED_TRACE(static_cast<int>(op));
                    SCOPED_TRACE(negate);
                    auto typed = std::make_shared<expr::NullExpr>(
                        expr::ColumnInfo(
                            field_id, DataType::ARRAY, element_type, {}, true),
                        op);
                    auto evaluate = [&](segcore::SegmentSealed* segment) {
                        auto leaf = std::make_shared<exec::PhyNullExpr>(
                            std::vector<exec::ExprPtr>{},
                            typed,
                            "PhyNullExpr",
                            nullptr,
                            segment,
                            kRows,
                            kRows,
                            0);
                        exec::ExprPtr physical = leaf;
                        if (negate) {
                            auto logical =
                                std::make_shared<expr::LogicalUnaryExpr>(
                                    expr::LogicalUnaryExpr::OpType::LogicalNot,
                                    typed);
                            physical =
                                std::make_shared<exec::PhyLogicalUnaryExpr>(
                                    std::vector<exec::ExprPtr>{leaf},
                                    logical,
                                    "PhyLogicalUnaryExpr",
                                    nullptr);
                        }
                        exec::QueryContext query_context(
                            "array-null", segment, kRows, MAX_TIMESTAMP);
                        exec::ExecContext exec_context(&query_context);
                        exec::EvalCtx context(&exec_context);
                        VectorPtr output;
                        physical->Eval(context, output);
                        return std::make_pair(
                            leaf,
                            std::dynamic_pointer_cast<ColumnVector>(output));
                    };
                    const auto [leaf, result] = evaluate(indexed.get());
                    const auto [raw_leaf, expected] = evaluate(raw.get());
                    EXPECT_EQ(leaf->GetExecPath(),
                              exec::ExprExecPath::ScalarIndex);
                    EXPECT_EQ(raw_leaf->GetExecPath(),
                              exec::ExprExecPath::RawData);
                    ASSERT_NE(result, nullptr);
                    ASSERT_NE(expected, nullptr);
                    ASSERT_EQ(result->size(), kRows);
                    ASSERT_EQ(expected->size(), kRows);
                    TargetBitmapView bits(result->GetRawData(), kRows);
                    TargetBitmapView valid(result->GetValidRawData(), kRows);
                    TargetBitmapView raw_bits(expected->GetRawData(), kRows);
                    TargetBitmapView raw_valid(expected->GetValidRawData(),
                                               kRows);
                    for (size_t row = 0; row < kRows; ++row) {
                        const bool is_null = row == 0 || row == 3;
                        const bool expected_bit =
                            (op == proto::plan::NullExpr_NullOp_IsNull
                                 ? is_null
                                 : !is_null) ^
                            negate;
                        EXPECT_TRUE(valid[row]) << row;
                        EXPECT_TRUE(raw_valid[row]) << row;
                        EXPECT_EQ(bits[row], raw_bits[row]) << row;
                        EXPECT_EQ(bits[row], expected_bit) << row;
                    }
                }
            }
            EXPECT_EQ(observed_pin, nullptr);
        }
    }
}

}  // namespace
}  // namespace milvus::test
