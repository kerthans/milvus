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

#include "exec/expression/ExprIndexIntegrationTestUtils.h"
#include "exec/expression/IndexPathSelection.h"
#include "exec/expression/UnaryExpr.h"
#include "index/Families.h"

namespace milvus::test {
namespace {

segcore::IndexCapabilityEntry
FlatEntry(int64_t id, std::string root, bool exact) {
    return {.key = {FieldId(100), segcore::IndexIdentity::PreBuiltIndex(id)},
            .json_path = std::move(root),
            .family = index::families::kJsonFlat,
            .value_type = DataType::JSON,
            .caps = {.json_paths = true, .exact = exact}};
}

TEST(JsonFlatPrefixSelectionTest, PreservesExactRefineAndCheapPriorities) {
    exec::ExprIndexRequirement req{.field_id = FieldId(100),
                                   .reader = exec::RequiredReader::Predicate,
                                   .value_type = DataType::INT64,
                                   .is_json_field = true,
                                   .json_path = "/profile/age",
                                   .accepts_candidates = true};
    auto choose = [&](std::vector<segcore::IndexCapabilityEntry> entries) {
        return exec::DetermineExecPath(
            req,
            segcore::FieldIndexCapability(req.field_id, std::move(entries)));
    };
    auto result =
        choose({FlatEntry(10, "", true), FlatEntry(20, "/profile", true)});
    ASSERT_TRUE(result.key.has_value());
    EXPECT_TRUE(result.key->identity ==
                segcore::IndexIdentity::PreBuiltIndex(20));

    // Exact path outranks every prefix even when it needs refinement.
    result = choose({FlatEntry(10, "/profile/age", false),
                     FlatEntry(20, "/profile", true)});
    ASSERT_TRUE(result.key.has_value());
    EXPECT_TRUE(result.key->identity ==
                segcore::IndexIdentity::PreBuiltIndex(10));

    // Root length only breaks a tie in the pre-existing score.
    result =
        choose({FlatEntry(10, "", true), FlatEntry(20, "/profile", false)});
    ASSERT_TRUE(result.key.has_value());
    EXPECT_TRUE(result.key->identity ==
                segcore::IndexIdentity::PreBuiltIndex(10));
    // JSON flat routers do not expose ValueLookup. Compare usable typed JSON
    // entries at the same exact path; cheap lookup must beat identity order.
    req.reader = exec::RequiredReader::ValueLookup;
    req.value_type = DataType::VARCHAR;
    segcore::IndexCapabilityEntry expensive{
        .key = {req.field_id, segcore::IndexIdentity::PreBuiltIndex(10)},
        .json_path = req.json_path,
        .json_cast_type = JsonCastType::FromString("VARCHAR"),
        .family = index::families::kBitmap,
        .value_type = DataType::VARCHAR,
        .caps = {.predicate = true, .value_lookup = true, .exact = true}};
    auto cheap = expensive;
    cheap.key.identity = segcore::IndexIdentity::PreBuiltIndex(20);
    cheap.family = index::families::kSort;
    cheap.caps.cheap_value_lookup = true;
    result = choose({expensive, cheap});
    ASSERT_TRUE(result.key.has_value());
    EXPECT_TRUE(result.key->identity ==
                segcore::IndexIdentity::PreBuiltIndex(20));
}

class SelectedRootUnary final : public exec::PhyUnaryRangeFilterExpr {
 public:
    using PhyUnaryRangeFilterExpr::PhyUnaryRangeFilterExpr;

    std::string
    SelectedRoot() const {
        return selected_index_entry_ ? selected_index_entry_->json_path
                                     : "<raw>";
    }
};

TEST(JsonFlatPrefixSelectionTest, PinsLongestRootAndMatchesRawInEitherIdOrder) {
    const std::vector<std::string> rows{
        R"({"profile":{"age":20},"sibling":"outside"})",
        R"({"profile":{"age":40}})",
        R"({"profile":{"age":null}})",
        R"({"profile":{}})",
        R"({"other":1})"};
    for (const bool narrow_first : {false, true}) {
        SCOPED_TRACE(narrow_first);
        auto schema = std::make_shared<Schema>();
        const auto field_id =
            schema->AddDebugField("json", DataType::JSON, true);
        expr_index::RawFieldFiles raw_files;
        auto field = expr_index::JsonField(rows, true);
        auto indexed = segcore::CreateSealedSegment(schema);
        auto raw = segcore::CreateSealedSegment(schema);
        const auto raw_info = raw_files.Prepare(field_id, {field});
        indexed->LoadFieldData(raw_info);
        raw->LoadFieldData(raw_info);
        OpContext untouched;
        OpContext* broad_pin = &untouched;
        OpContext* narrow_pin = &untouched;
        auto install = [&](const std::string& root, OpContext** observed) {
            auto opened = expr_index::BuildIndex(
                field_id,
                DataType::JSON,
                index::INVERTED_INDEX_TYPE,
                {field},
                {{JSON_PATH, root}, {JSON_CAST_TYPE, "JSON"}});
            expr_index::InstallIndex(*indexed,
                                     field_id,
                                     DataType::JSON,
                                     std::move(opened),
                                     DataType::NONE,
                                     observed);
        };
        if (narrow_first) {
            install("/profile", &narrow_pin);
            install("", &broad_pin);
        } else {
            install("", &broad_pin);
            install("/profile", &narrow_pin);
        }
        const auto capability = indexed->IndexCapability(field_id);
        ASSERT_EQ(capability.entries().size(), 2);
        // InstallIndex allocates increasing ids; the first iteration puts the
        // broad root first in identity order, opposite the required selection.
        EXPECT_EQ(capability.entries()[0].json_path,
                  narrow_first ? "/profile" : "");

        proto::plan::GenericValue value;
        value.set_int64_val(30);
        auto typed = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(
                field_id, DataType::JSON, {"profile", "age"}, true),
            proto::plan::GreaterThan,
            value);
        auto evaluate = [&](segcore::SegmentSealed* segment) {
            auto physical = std::make_shared<SelectedRootUnary>(
                std::vector<exec::ExprPtr>{},
                typed,
                "PhyUnaryRangeFilterExpr",
                nullptr,
                segment,
                rows.size(),
                rows.size(),
                0);
            exec::QueryContext query_context(
                "json-prefix", segment, rows.size(), MAX_TIMESTAMP);
            exec::ExecContext exec_context(&query_context);
            exec::EvalCtx context(&exec_context);
            VectorPtr result;
            physical->Eval(context, result);
            return std::make_pair(
                physical, std::dynamic_pointer_cast<ColumnVector>(result));
        };
        const auto [physical, actual] = evaluate(indexed.get());
        const auto [raw_physical, expected] = evaluate(raw.get());
        ASSERT_NE(actual, nullptr);
        ASSERT_NE(expected, nullptr);
        EXPECT_EQ(physical->GetExecPath(), exec::ExprExecPath::ScalarIndex);
        EXPECT_EQ(physical->SelectedRoot(), "/profile");
        EXPECT_EQ(raw_physical->GetExecPath(), exec::ExprExecPath::RawData);
        EXPECT_EQ(broad_pin, &untouched);
        EXPECT_EQ(narrow_pin, nullptr);
        TargetBitmapView bits(actual->GetRawData(), actual->size());
        TargetBitmapView valid(actual->GetValidRawData(), actual->size());
        TargetBitmapView raw_bits(expected->GetRawData(), expected->size());
        TargetBitmapView raw_valid(expected->GetValidRawData(),
                                   expected->size());
        ASSERT_EQ(actual->size(), rows.size());
        ASSERT_EQ(expected->size(), rows.size());
        for (size_t row = 0; row < rows.size(); ++row) {
            EXPECT_EQ(valid[row], raw_valid[row]) << row;
            EXPECT_EQ(bits[row] && valid[row], raw_bits[row] && raw_valid[row])
                << row;
            EXPECT_EQ(bits[row] && valid[row], row == 1) << row;
        }
    }
}

}  // namespace
}  // namespace milvus::test
