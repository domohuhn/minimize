// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#include "minimize/bootstrap.hpp"

#include <cmath>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "common.hpp"

using Catch::Approx;
using namespace minimize;

SCENARIO("Statistics data", "[bootstrap]") {
    GIVEN("a vector of parameters") {
        std::vector<minimize::parameter_t<4>> values;
        constexpr double deviation = 0.25;
        for (int i = 0; i < 3 * 128; ++i) {
            const auto var = (i % 3 - 1) * deviation;
            values.emplace_back(minimize::parameter_t<4>{16.0 + var, -3.0 + 2 * var, 32.0 + 3 * var, -27.0 + 4 * var});
        }
        WHEN("the mean is computed") {
            const auto mean = detail::compute_mean(values);
            THEN("the correct value is returned") {
                REQUIRE_THAT(mean[0], Catch::Matchers::WithinRel(16.0, 1e-16));
                REQUIRE_THAT(mean[1], Catch::Matchers::WithinRel(-3.0, 1e-16));
                REQUIRE_THAT(mean[2], Catch::Matchers::WithinRel(32.0, 1e-16));
                REQUIRE_THAT(mean[3], Catch::Matchers::WithinRel(-27.0, 1e-16));
            }
        }

        WHEN("the stddev is computed") {
            const auto stddev = detail::compute_stddev(values);
            THEN("the correct value is returned") {
                const auto expected = std::sqrt(2.0 * deviation * deviation / 3.0);
                REQUIRE_THAT(stddev[0], Catch::Matchers::WithinRel(expected, 1e-16));
                REQUIRE_THAT(stddev[1], Catch::Matchers::WithinRel(2.0 * expected, 1e-16));
                REQUIRE_THAT(stddev[2], Catch::Matchers::WithinRel(3.0 * expected, 1e-16));
                REQUIRE_THAT(stddev[3], Catch::Matchers::WithinRel(4.0 * expected, 1e-16));
            }
        }
    }
}
