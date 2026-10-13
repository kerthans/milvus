// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file to you under
// the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License
// at
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
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/Geometry.h"
#include "index/contracts/query/INullReader.h"
#include "index/contracts/query/ISpatialReader.h"
#include "index/growing/RTreeGrowingSpatialIndex.h"
#include "index/scalar/spatial/RTreeIndexReader.h"
#include "index/test_utils/AssertHelpers.h"
#include "index/test_utils/CaseTestDriver.h"

namespace milvus::index::test {

class RTreeGrowingSpatialIndexTest : public ::testing::Test {
 protected:
    static const SharedValidityBitmap::Snapshot&
    Validity(const GrowingIndexSnapshotPin& pin) {
        const auto& reader =
            dynamic_cast<const RTreeIndexReader&>(pin.Reader());
        return reader.Validity();
    }

    static void
    AppendNulls(RTreeGrowingSpatialIndex& owner,
                size_t row_begin,
                size_t count) {
        const std::vector<std::string_view> values(count);
        const auto valid = std::make_unique<bool[]>(count);
        owner.Append(static_cast<int64_t>(row_begin),
                     {count, values.data(), valid.get()});
    }

    static void
    ExpectValidity(const GrowingIndexSnapshotPin& pin,
                   const std::vector<uint8_t>& valid) {
        ASSERT_TRUE(static_cast<bool>(pin));
        EXPECT_EQ(pin.CoveredRowEnd(), valid.size());
        EXPECT_EQ(pin.Reader().Count(), valid.size());
        const auto& nulls = dynamic_cast<const INullReader&>(pin.Reader());
        const auto is_null = nulls.IsNull();
        const auto is_not_null = nulls.IsNotNull();
        // Invalid-query fallback is also the spatial reader's validity path.
        const Geometry invalid_query;
        const auto candidates =
            dynamic_cast<const ISpatialReader&>(pin.Reader())
                .Candidates(SpatialOp::Intersects, invalid_query);
        ASSERT_EQ(is_null.size(), valid.size());
        ASSERT_EQ(is_not_null.size(), valid.size());
        ASSERT_EQ(candidates.size(), valid.size());
        for (size_t row = 0; row < valid.size(); ++row) {
            EXPECT_EQ(static_cast<bool>(is_null[row]), valid[row] == 0) << row;
            EXPECT_EQ(static_cast<bool>(is_not_null[row]), valid[row] != 0)
                << row;
            EXPECT_EQ(static_cast<bool>(candidates[row]), valid[row] != 0)
                << row;
        }
    }
};

namespace {

TEST_F(RTreeGrowingSpatialIndexTest,
       PreservesNullableBitmapsAcrossPublications) {
    RTreeGrowingSpatialIndex owner(4);
    const std::string point =
        Geometry(GetThreadLocalGEOSContext(), "POINT(1 1)").to_wkb_string();
    const std::array<std::string_view, 4> values{point, "", point, ""};
    const std::array<bool, 4> first_valid{true, false, true, false};
    owner.Append(0, {values.size(), values.data(), first_valid.data()});
    const auto first = owner.PinSnapshot();
    ExpectValidity(first, {1, 0, 1, 0});

    const std::array<bool, 4> second_valid{false, true, true, false};
    owner.Append(4, {values.size(), values.data(), second_valid.data()});
    const auto second = owner.PinSnapshot();
    ExpectValidity(second, {1, 0, 1, 0, 0, 1, 1, 0});
    ASSERT_NE(Validity(first).FullWords().data(), nullptr);
    EXPECT_EQ(Validity(first).FullWords().data(),
              Validity(second).FullWords().data());
    EXPECT_EQ(Validity(first).BufferBytes(), 128 * 1024);

    owner.Append(8, {values.size(), values.data(), nullptr});
    const auto third = owner.PinSnapshot();
    ExpectValidity(third, {1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1});
    ExpectValidity(first, {1, 0, 1, 0});
    ExpectValidity(second, {1, 0, 1, 0, 0, 1, 1, 0});
}

TEST_F(RTreeGrowingSpatialIndexTest,
       SharesValidityWordsAcrossBoundaryAndOwnerLifetime) {
    constexpr size_t kWordBits = 64;
    auto owner = std::make_unique<RTreeGrowingSpatialIndex>(16);
    AppendNulls(*owner, 0, kWordBits - 1);
    owner->Flush();
    const auto partial = owner->PinSnapshot();
    const std::vector<uint8_t> partial_valid(kWordBits - 1, 0);
    const std::string point =
        Geometry(GetThreadLocalGEOSContext(), "POINT(1 1)").to_wkb_string();
    const std::array<std::string_view, 3> values{point, "", point};
    const std::array<bool, 3> crossing_valid{true, false, true};
    owner->Append(kWordBits - 1,
                  {values.size(), values.data(), crossing_valid.data()});
    owner->Flush();
    const auto full = owner->PinSnapshot();
    auto full_valid = partial_valid;
    full_valid.insert(full_valid.end(), {1, 0, 1});
    EXPECT_EQ(Validity(partial).FullWords().data(),
              Validity(full).FullWords().data());
    const std::array<bool, 3> tail_valid{false, true, false};
    owner->Append(kWordBits + 2,
                  {values.size(), values.data(), tail_valid.data()});
    owner->Flush();
    const auto tail = owner->PinSnapshot();
    auto tail_expected = full_valid;
    tail_expected.insert(tail_expected.end(), {0, 1, 0});
    EXPECT_EQ(Validity(full).FullWords().data(),
              Validity(tail).FullWords().data());
    owner.reset();
    ExpectValidity(partial, partial_valid);
    ExpectValidity(full, full_valid);
    ExpectValidity(tail, tail_expected);
}

TEST_F(RTreeGrowingSpatialIndexTest, AcceptsSeveralValidityWordsInOneAppend) {
    constexpr size_t kWordBits = 64;
    RTreeGrowingSpatialIndex owner(16);
    AppendNulls(owner, 0, 2 * kWordBits + 7);
    const auto first = owner.PinSnapshot();
    AppendNulls(owner, 2 * kWordBits + 7, kWordBits - 7);
    const auto second = owner.PinSnapshot();
    EXPECT_EQ(Validity(first).FullWords().data(),
              Validity(second).FullWords().data());
    ExpectValidity(first, std::vector<uint8_t>(2 * kWordBits + 7, 0));
    ExpectValidity(second, std::vector<uint8_t>(3 * kWordBits, 0));
}

TEST_F(RTreeGrowingSpatialIndexTest, AllValidPrefixAllocatesOnlyAtFirstNull) {
    RTreeGrowingSpatialIndex owner(16);
    const std::vector<std::string_view> values(65);
    owner.Append(0, {values.size(), values.data(), nullptr});
    const auto all_valid = owner.PinSnapshot();
    EXPECT_TRUE(Validity(all_valid).AllValid());
    EXPECT_EQ(Validity(all_valid).BufferBytes(), 0);
    AppendNulls(owner, 65, 1);
    owner.Flush();
    const auto nullable = owner.PinSnapshot();
    EXPECT_FALSE(Validity(nullable).AllValid());
    ExpectValidity(all_valid, std::vector<uint8_t>(65, 1));
    auto expected = std::vector<uint8_t>(65, 1);
    expected.push_back(0);
    ExpectValidity(nullable, expected);
}

TEST(RTreeSealedValidityTest, LoadsNullableAndAllValidSnapshots) {
    for (const bool nullable : {false, true}) {
        for (const auto backend_name :
             nullable
                 ? std::array{"SpatialRTreeHeap", "SpatialRTreeMmapRequested"}
                 : std::array{"SpatialRTreeHeapNonNull",
                              "SpatialRTreeMmapRequestedNonNull"}) {
            const auto& backend =
                ScalarReaderBackends().Get<std::string_view>(backend_name);
            const auto& dataset = ScalarDataSets().Get<std::string_view>(
                nullable ? "SpatialWkbNullable" : "SpatialWkbAllValid");
            auto data = dataset.make_data();
            const ScalarTestInput<std::string_view> input(data);
            auto artifact =
                backend.Build(input.View(), {.row_count = data.values.size()});
            auto reader = backend.Open(std::move(artifact),
                                       {.row_count = data.values.size()});
            const auto* spatial =
                dynamic_cast<const RTreeIndexReader*>(reader.get());
            ASSERT_NE(spatial, nullptr);
            EXPECT_EQ(spatial->Validity().AllValid(), !nullable);
            const auto is_null = spatial->IsNull();
            const auto is_not_null = spatial->IsNotNull();
            ASSERT_EQ(is_null.size(), data.values.size());
            for (size_t row = 0; row < data.values.size(); ++row) {
                const bool valid = !data.validity_present || data.validity[row];
                EXPECT_EQ(is_null[row], !valid) << row;
                EXPECT_EQ(is_not_null[row], valid) << row;
            }
            if (!nullable) {
                EXPECT_EQ(spatial->Validity().BufferBytes(), 0);
            }
        }
    }
}

constexpr auto kBackend = "SpatialRTreeHeap";
constexpr auto kDataset = "SpatialWkbNullable";
constexpr std::array<std::string_view, 4> kAllBackends = {
    "SpatialRTreeHeap",
    "SpatialRTreeHeapNonNull",
    "SpatialRTreeMmapRequested",
    "SpatialRTreeMmapRequestedNonNull",
};

struct ConcreteSpatialCase {
    std::string name;
    SpatialOp op;
    std::optional<std::string> query_wkt;
    std::vector<size_t> expected_offsets;
};

const IndexTestCases&
ConcreteRTreeCases() {
    static const auto cases = [] {
        IndexTestCases cases;
        const std::vector<ConcreteSpatialCase> table = {
            // Offsets 8 (GEOMETRYCOLLECTION EMPTY) and 9 (corrupt WKB) are
            // non-null rows with no computable envelope, so they carry the
            // deterministic placeholder MBR at the origin and this
            // origin-covering query pulls them into the candidate set. Exact
            // refinement discards them; dropping them from the index instead
            // would desynchronize the index row count from the segment row
            // count. Regression guard for the placeholder contract (#50951).
            {.name = "PointMbrIntersection",
             .op = SpatialOp::Equals,
             .query_wkt = "POINT(0 0)",
             .expected_offsets = {0, 2, 3, 6, 8, 9}},
            // The same two rows stay out of every candidate set whose box does
            // not cover the origin, which is what keeps the placeholder cheap.
            {.name = "PlaceholderRowsStayOutOfDistantMbr",
             .op = SpatialOp::Intersects,
             .query_wkt = "POLYGON((4 4,4 6,6 6,6 4,4 4))",
             .expected_offsets = {5, 7}},
            {.name = "OperatorIndependentMbrIntersection",
             .op = SpatialOp::Contains,
             .query_wkt = "POINT(1 1)",
             .expected_offsets = {2, 3, 4, 11}},
            {.name = "CallerExpandedDWithinMbr",
             .op = SpatialOp::DWithin,
             .query_wkt = "POLYGON((1.5 1.5,1.5 2.5,2.5 2.5,2.5 1.5,1.5 1.5))",
             .expected_offsets = {1, 2, 3, 4}},
            {.name = "InvalidQueryFallsBackToEveryNonNullRow",
             .op = SpatialOp::Intersects,
             .query_wkt = std::nullopt,
             .expected_offsets = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11}},
        };
        for (const auto& test_case : table) {
            cases.Add(IndexTestCase<std::string_view>{
                .name = test_case.name,
                .dataset = kDataset,
                .input_shape = BackendInputShape::SpatialWkb,
                .domain = Domain::Row,
                .logical_value_type = DataType::GEOMETRY,
                .input_lifetime = InputLifetime::ReleaseBeforeBody,
                .backends = {kBackend},
                .body =
                    Observe<std::string_view>{
                        .capability = &ReaderCaps::spatial,
                        .run =
                            [test_case](
                                const ReaderBackend&,
                                const ScalarTestData<std::string_view>& data,
                                IIndexReaderBasePtr& reader) {
                                const auto* spatial =
                                    dynamic_cast<const ISpatialReader*>(
                                        reader.get());
                                ASSERT_NE(spatial, nullptr);
                                auto actual = [&] {
                                    if (test_case.query_wkt.has_value()) {
                                        const Geometry query(
                                            GetThreadLocalGEOSContext(),
                                            test_case.query_wkt->c_str());
                                        return spatial->Candidates(test_case.op,
                                                                   query);
                                    }
                                    const Geometry query;
                                    return spatial->Candidates(test_case.op,
                                                               query);
                                }();
                                const auto expected =
                                    Hits(data.values.size(),
                                         test_case.expected_offsets);
                                ExpectBitmap(actual, expected);
                            },
                    },
            });
        }
        cases.Add(IndexTestCase<std::string_view>{
            .name = "UsesHeapResourceAccounting",
            .dataset = "SpatialWkbAllValid",
            .input_shape = BackendInputShape::SpatialWkb,
            .domain = Domain::Row,
            .logical_value_type = DataType::GEOMETRY,
            .input_lifetime = InputLifetime::ReleaseBeforeBody,
            .backends = std::vector<std::string>(kAllBackends.begin(),
                                                 kAllBackends.end()),
            .body =
                Observe<std::string_view>{
                    .capability = &ReaderCaps::spatial,
                    .run =
                        [](const ReaderBackend&,
                           const ScalarTestData<std::string_view>&,
                           IIndexReaderBasePtr& reader) {
                            EXPECT_EQ(reader->CellByteSize().file_bytes, 0);
                        },
                },
        });
        return cases;
    }();
    return cases;
}

std::vector<FilterParam>
RTreeEmptyBuildCases() {
    const auto* dataset =
        &ScalarDataSets().Get<std::string_view>("SpatialEmpty");
    std::vector<FilterParam> cases;
    cases.reserve(kAllBackends.size());
    for (const auto name : kAllBackends) {
        const auto backend =
            ScalarReaderBackends().Get<std::string_view>(std::string(name));
        cases.push_back({
            .name = backend.Name() + "_SpatialEmpty_BuildRejectsEmptyInput",
            .run =
                [backend, dataset] {
                    auto data = dataset->make_data();
                    const ScalarTestInput<std::string_view> input(data);
                    try {
                        static_cast<void>(backend.Build(
                            input.View(),
                            {.row_count = 0, .values = data.metadata}));
                        ADD_FAILURE()
                            << "R-Tree empty build unexpectedly succeeded";
                    } catch (const SegcoreError& error) {
                        EXPECT_EQ(error.get_error_code(),
                                  ErrorCode::DataIsEmpty);
                    } catch (const std::exception& error) {
                        ADD_FAILURE()
                            << "R-Tree empty build threw a non-SegcoreError: "
                            << error.what();
                    } catch (...) {
                        ADD_FAILURE()
                            << "R-Tree empty build threw a non-SegcoreError";
                    }
                },
        });
    }
    return cases;
}

class RTreeIndexReaderTest : public ::testing::TestWithParam<FilterParam> {};

TEST_P(RTreeIndexReaderTest, ReturnsCurrentMbrCandidates) {
    GetParam().run();
}

INSTANTIATE_TEST_SUITE_P(RTreeHeap,
                         RTreeIndexReaderTest,
                         ::testing::ValuesIn(ConcreteRTreeCases().All()),
                         FilterParamName);

class RTreeIndexBuildTest : public ::testing::TestWithParam<FilterParam> {};

TEST_P(RTreeIndexBuildTest, RejectsEmptyInput) {
    GetParam().run();
}

INSTANTIATE_TEST_SUITE_P(RTreeProfiles,
                         RTreeIndexBuildTest,
                         ::testing::ValuesIn(RTreeEmptyBuildCases()),
                         FilterParamName);

}  // namespace
}  // namespace milvus::index::test
