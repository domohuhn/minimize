// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#include "minimize/find_minimum_on_line.hpp"

#include <cmath>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "common.hpp"

using Catch::Approx;
using namespace minimize;

SCENARIO("Minimum along a gradient can be found", "[line search]") {
    GIVEN("Perfect data") {
        SaddleFunction saddle{};

        // measurements for 0.5, 1.0, 1.3, 5.0
        MeasurementVector<2> vec = create_perfect_test_data_saddle();

        parameter_t<4> direction{-1.0, -2.0, -2.6, -10.0};

        WHEN("an interval with the minimum inside is searched along the gradient") {
            const auto interval = search_interval_around_minimum(saddle, vec, direction, 100);
            THEN("the real minimum is bracketed") {
                REQUIRE(interval.before[0] < 0.5);
                REQUIRE(interval.before[1] < 1.0);
                REQUIRE(interval.before[2] < 1.3);
                REQUIRE(interval.before[3] < 5.0);

                REQUIRE(interval.past[0] > 0.5);
                REQUIRE(interval.past[1] > 1.0);
                REQUIRE(interval.past[2] > 1.3);
                REQUIRE(interval.past[3] > 5.0);
            }
        }

        WHEN("the minimum in an interval is searched with a symmetric interval") {
            parameter_t<4> lower{0.25, 0.5, 0.65, 2.5};
            parameter_t<4> upper{0.75, 1.5, 1.95, 7.5};
            auto found = binary_search_minimum_in_interval(saddle, vec, lower, upper, 100);
            THEN("the real minimum is found") {
                REQUIRE(found[0] == 0.5);
                REQUIRE(found[1] == 1.0);
                REQUIRE(found[2] == 1.3);
                REQUIRE(found[3] == 5.0);
            }
        }

        WHEN("the minimum in an interval is searched with an asymmetric interval 1") {
            parameter_t<4> lower{0.25, 0.5, 0.65, 2.5};
            parameter_t<4> upper{0.505, 1.01, 1.313, 5.05};
            auto found = binary_search_minimum_in_interval(saddle, vec, lower, upper, 100);
            THEN("the real minimum is found") {
                REQUIRE(found[0] == 0.5);
                REQUIRE(found[1] == 1.0);
                REQUIRE(found[2] == 1.3);
                REQUIRE(found[3] == 5.0);
            }
        }

        WHEN("the minimum in an interval is searched with an asymmetric interval 2") {
            parameter_t<4> lower{0.49, 0.98, 1.274, 4.9};
            parameter_t<4> upper{0.75, 1.5, 1.95, 7.5};
            auto found = binary_search_minimum_in_interval(saddle, vec, lower, upper, 100);
            THEN("the real minimum is found") {
                REQUIRE(found[0] == 0.5);
                REQUIRE(found[1] == 1.0);
                REQUIRE(found[2] == 1.3);
                REQUIRE(found[3] == 5.0);
            }
        }

        WHEN("the minimum is searched") {
            auto found = find_minimum_on_line(saddle, saddle.parameters(), vec, direction, 200);
            THEN("the real minimum is found") {
                REQUIRE(found[0] == Approx(0.5));
                REQUIRE(found[1] == Approx(1.0));
                REQUIRE(found[2] == Approx(1.3));
                REQUIRE(found[3] == Approx(5.0));
            }
        }
    }

    GIVEN("Noisy data") {
        SaddleFunction saddle{};

        // measurements for 0.5, 1.0, 1.3, 5.0
        MeasurementVector<2> vec = create_noisy_test_data_saddle();
        parameter_t<4> direction{-1.0, -2.0, -2.6, -10.0};

        WHEN("an interval with the minimum inside is searched along the gradient") {
            const auto interval = search_interval_around_minimum(saddle, vec, direction, 100);
            THEN("the real minimum is bracketed") {
                REQUIRE(interval.before[0] < 0.5);
                REQUIRE(interval.before[1] < 1.0);
                REQUIRE(interval.before[2] < 1.3);
                REQUIRE(interval.before[3] < 5.0);

                REQUIRE(interval.past[0] > 0.5);
                REQUIRE(interval.past[1] > 1.0);
                REQUIRE(interval.past[2] > 1.3);
                REQUIRE(interval.past[3] > 5.0);
            }
        }

        WHEN("the minimum is searched along the gradient") {
            parameter_t<4> lower{0.25, 0.5, 0.65, 2.5};
            parameter_t<4> upper{0.75, 1.5, 1.95, 7.5};
            auto found = binary_search_minimum_in_interval(saddle, vec, lower, upper, 100);
            THEN("the real minimum is found") {
                REQUIRE_THAT(found[0], Catch::Matchers::WithinRel(0.5, 0.0001));
                REQUIRE_THAT(found[1], Catch::Matchers::WithinRel(1.0, 0.0001));
                REQUIRE_THAT(found[2], Catch::Matchers::WithinRel(1.3, 0.0001));
                REQUIRE_THAT(found[3], Catch::Matchers::WithinRel(5.0, 0.0001));
            }
        }

        WHEN("the minimum is searched") {
            auto found = find_minimum_on_line(saddle, saddle.parameters(), vec, direction, 100);
            THEN("the real minimum is found") {
                REQUIRE_THAT(found[0], Catch::Matchers::WithinRel(0.5, 0.0001));
                REQUIRE_THAT(found[1], Catch::Matchers::WithinRel(1.0, 0.0001));
                REQUIRE_THAT(found[2], Catch::Matchers::WithinRel(1.3, 0.0001));
                REQUIRE_THAT(found[3], Catch::Matchers::WithinRel(5.0, 0.0001));
            }
        }
    }
}
