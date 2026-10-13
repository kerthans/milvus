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
#include <barrier>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#include "index/SharedValidityBitmap.h"

namespace milvus::index::test {
namespace {

void
ExpectSnapshot(const SharedValidityBitmap::Snapshot& snapshot,
               std::span<const size_t> nulls) {
    auto actual = snapshot.ToBitmap();
    TargetBitmap expected(snapshot.Count(), true);
    for (const auto row : nulls) {
        expected.reset(row);
    }
    const auto check = [&] {
        ASSERT_EQ(actual.size(), expected.size());
        EXPECT_EQ(actual.count(), expected.count());
        if (actual.size_in_bytes() != 0) {
            EXPECT_EQ(
                std::memcmp(
                    actual.data(), expected.data(), actual.size_in_bytes()),
                0);
        }
        for (size_t row = 0; row < expected.size(); ++row) {
            EXPECT_EQ(actual[row], expected[row]) << row;
        }
    };
    check();
    actual.flip();
    expected.flip();
    check();
}

TEST(SharedValidityBitmapTest, AllValidPrefixesDoNotAllocate) {
    SharedValidityBitmap writer(64);
    for (const auto count : {size_t{0}, size_t{1}, size_t{65}, size_t{513}}) {
        writer.PrepareAppend(count, false);
        writer.FinishAppend(count);
        const auto pin = writer.Freeze();
        EXPECT_TRUE(pin.AllValid());
        EXPECT_EQ(pin.BufferBytes(), 0);
        EXPECT_EQ(pin.FullWords().data(), nullptr);
        ExpectSnapshot(pin, {});
    }
}

TEST(SharedValidityBitmapTest, FirstNullAtZeroOrAfterAnAllValidPrefix) {
    for (const size_t first_null : {size_t{0}, size_t{17}}) {
        SharedValidityBitmap writer(64);
        writer.PrepareAppend(first_null, false);
        writer.FinishAppend(first_null);
        const auto before = writer.Freeze();
        writer.PrepareAppend(first_null + 1, true);
        writer.AppendNull(first_null);
        writer.FinishAppend(first_null + 1);
        const auto after = writer.Freeze();
        EXPECT_TRUE(before.AllValid());
        EXPECT_FALSE(after.AllValid());
        EXPECT_EQ(after.BufferBytes(), sizeof(uint64_t));
        ExpectSnapshot(before, {});
        const std::array<size_t, 1> nulls{first_null};
        ExpectSnapshot(after, nulls);
    }
}

TEST(SharedValidityBitmapTest, WordAndCapacityBoundariesPreserveOldSnapshots) {
    SharedValidityBitmap writer(128);
    writer.PrepareAppend(64, true);
    writer.AppendNull(63);
    writer.FinishAppend(64);
    const auto at_word = writer.Freeze();
    writer.PrepareAppend(65, true);
    writer.AppendNull(64);
    writer.FinishAppend(65);
    const auto after_word = writer.Freeze();
    writer.PrepareAppend(128, true);
    writer.AppendNull(127);
    writer.FinishAppend(128);
    const auto at_capacity = writer.Freeze();
    EXPECT_EQ(at_word.FullWords().data(), after_word.FullWords().data());
    EXPECT_EQ(after_word.FullWords().data(), at_capacity.FullWords().data());
    EXPECT_EQ(at_capacity.BufferBytes(), 16);
    writer.PrepareAppend(129, true);
    writer.AppendNull(128);
    writer.FinishAppend(129);
    const auto grown = writer.Freeze();
    EXPECT_NE(at_capacity.FullWords().data(), grown.FullWords().data());
    EXPECT_EQ(grown.BufferBytes(), 32);
    const std::array<size_t, 4> nulls{63, 64, 127, 128};
    ExpectSnapshot(at_word, std::span(nulls).first(1));
    ExpectSnapshot(after_word, std::span(nulls).first(2));
    ExpectSnapshot(at_capacity, std::span(nulls).first(3));
    ExpectSnapshot(grown, nulls);
}

TEST(SharedValidityBitmapTest, PrivateTailAndPaddingIgnoreLaterClearedBits) {
    SharedValidityBitmap writer(128);
    writer.PrepareAppend(1, true);
    writer.AppendNull(0);
    writer.FinishAppend(1);
    const auto partial = writer.Freeze();
    writer.PrepareAppend(64, true);
    writer.AppendNull(17);
    writer.AppendNull(63);
    writer.FinishAppend(64);
    const auto complete = writer.Freeze();
    EXPECT_EQ(partial.FullWords().data(), complete.FullWords().data());
    const std::array<size_t, 3> nulls{0, 17, 63};
    ExpectSnapshot(partial, std::span(nulls).first(1));
    ExpectSnapshot(complete, nulls);
}

TEST(SharedValidityBitmapTest,
     CapacityCoversBatchEndAndPreparationDoesNotAccept) {
    SharedValidityBitmap writer(64);
    writer.PrepareAppend(513, true);
    EXPECT_EQ(writer.Freeze().Count(), 0);
    writer.AppendNull(512);
    writer.FinishAppend(513);
    const auto pin = writer.Freeze();
    EXPECT_GE(pin.BufferBytes() * 8, 513);
    const std::array<size_t, 1> nulls{512};
    ExpectSnapshot(pin, nulls);
    // Simulate a prepared batch rejected by its engine: no NULLs/count recorded.
    writer.PrepareAppend(1025, true);
    const auto after_rejection = writer.Freeze();
    EXPECT_EQ(after_rejection.Count(), 513);
    ExpectSnapshot(after_rejection, nulls);
    ExpectSnapshot(pin, nulls);
}

TEST(SharedValidityBitmapTest, RejectsOverflowAndShrinkingBeforeAcceptance) {
    EXPECT_THROW(SharedValidityBitmap(std::numeric_limits<size_t>::max()),
                 SegcoreError);
    EXPECT_THROW(SharedValidityBitmap(0), SegcoreError);
    SharedValidityBitmap writer(64);
    EXPECT_THROW(writer.PrepareAppend(std::numeric_limits<size_t>::max(), true),
                 SegcoreError);
    writer.PrepareAppend(65, true);
    writer.AppendNull(64);
    writer.FinishAppend(65);
    EXPECT_THROW(writer.PrepareAppend(64, true), SegcoreError);
    EXPECT_THROW(
        writer.PrepareAppend(std::numeric_limits<size_t>::max(), false),
        SegcoreError);
    const std::array<size_t, 1> nulls{64};
    ExpectSnapshot(writer.Freeze(), nulls);
}

TEST(SharedValidityBitmapTest, ConcurrentAppendNeverTouchesSnapshotTail) {
    SharedValidityBitmap writer(128);
    writer.PrepareAppend(65, true);
    writer.AppendNull(0);
    writer.AppendNull(64);
    writer.FinishAppend(65);
    const auto pin = writer.Freeze();
    const std::array<size_t, 2> nulls{0, 64};
    std::barrier start(2);
    std::thread append([&] {
        start.arrive_and_wait();
        for (size_t row = 65; row < 2048; ++row) {
            writer.PrepareAppend(row + 1, true);
            writer.AppendNull(row);
            writer.FinishAppend(row + 1);
        }
    });
    start.arrive_and_wait();
    for (size_t query = 0; query < 256; ++query) {
        ExpectSnapshot(pin, nulls);
    }
    append.join();
    ExpectSnapshot(pin, nulls);
}

}  // namespace
}  // namespace milvus::index::test
