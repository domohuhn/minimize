// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#include <cmath>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "minimize/function.hpp"

SCENARIO("Polynomials", "[function]") {
    GIVEN("A polynomial of first degree") {
        minimize::Polynomial<1> poly{{-2.0, 3.0}};

        WHEN("a value is computed") {
            const auto a = poly.evaluate(0.0);
            const auto b = poly.evaluate(-1.0);
            const auto c = poly.evaluate(2.0);
            THEN("the result is correct") {
                REQUIRE(a == -2.0);
                REQUIRE(b == -5.0);
                REQUIRE(c == 4.0);
            }
        }
    }

    GIVEN("A polynomial of second degree") {
        minimize::Polynomial<2> poly{{-2.0, 3.0, 1.0}};

        WHEN("a value is computed") {
            const auto a = poly.evaluate(0.0);
            const auto b = poly.evaluate(-1.0);
            const auto c = poly.evaluate(2.0);
            const auto d = poly.evaluate(1.0);
            THEN("the result is correct") {
                REQUIRE(a == -2.0);
                REQUIRE(b == -4.0);
                REQUIRE(c == 8.0);
                REQUIRE(d == 2.0);
            }
        }
    }
}
