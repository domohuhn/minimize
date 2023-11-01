// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#include "minimize/chi2.hpp"

#include <cmath>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common.hpp"

using Catch::Approx;
using namespace minimize;

SCENARIO("chi2 can be computed", "[function]") {
    GIVEN("A linear function") {
        LinearFunction linear{};
        WHEN("chi2 is computed with exact values") {
            MeasurementVector<1> vec{};
            for (size_t i = 0; i < 10; ++i) {
                double in{static_cast<double>(i)};
                vec.push_back(Measurement<1>{in, linear.evaluate(in)});
            }
            THEN("the result is zero") { REQUIRE(compute_chi2(linear, vec) == Approx(0.0)); }
        }
        WHEN("chi2 is computed with a offset") {
            MeasurementVector<1> vec{};
            for (size_t i = 0; i < 10; ++i) {
                double in{static_cast<double>(i)};
                vec.push_back(Measurement<1>{in, linear.evaluate(in) + 0.5});
            }
            THEN("the result is 2.5") { REQUIRE(compute_chi2(linear, vec) == Approx(2.5)); }
        }
    }
}

SCENARIO("chi2 gradient can be computed", "[function]") {
    GIVEN("A linear function") {
        WHEN("chi2 gradient is computed with a positive offset") {
            LinearFunction linear{};
            MeasurementVector<1> vec{};
            const double offset = 0.1;
            for (size_t i = 0; i < 10; ++i) {
                double in{static_cast<double>(i)};
                vec.push_back(Measurement<1>{in, linear.evaluate(in) + offset});
            }
            const auto grad = compute_chi2_gradient(linear, vec);
            THEN("the gradient points to negative values") {
                REQUIRE(grad[0] == Approx(-9.0));
                REQUIRE(grad[1] == Approx(-2.0));
            }
        }
    }
}

SCENARIO("chi2 can be computed with weights", "[function]") {
    GIVEN("A linear function") {
        LinearFunction linear{};
        WHEN("chi2 is computed with exact values") {
            MeasurementVectorWithErrors<1> vec{};
            for (size_t i = 0; i < 10; ++i) {
                double in{static_cast<double>(i)};
                vec.push_back(MeasurementWithError<1>{in, linear.evaluate(in), 0.5});
            }
            THEN("the result is zero") { REQUIRE(compute_chi2(linear, vec) == Approx(0.0)); }
        }

        WHEN("chi2 is computed with a offset") {
            MeasurementVectorWithErrors<1> vec{};
            for (size_t i = 0; i < 10; ++i) {
                double in{static_cast<double>(i)};
                vec.push_back(MeasurementWithError<1>{in, linear.evaluate(in) + 0.5, 0.5});
            }
            THEN("the result is 10") { REQUIRE(compute_chi2(linear, vec) == Approx(10.0)); }
        }
    }
}

SCENARIO("chi2 gradient can be computed with weights", "[function]") {
    GIVEN("A linear function") {
        WHEN("chi2 gradient is computed with a positive offset") {
            LinearFunction linear{};
            MeasurementVectorWithErrors<1> vec{};
            const double offset = 0.1;
            for (size_t i = 0; i < 10; ++i) {
                double in{static_cast<double>(i)};
                vec.push_back(MeasurementWithError<1>{in, linear.evaluate(in) + offset, 0.5});
            }
            const auto grad = compute_chi2_gradient(linear, vec);
            THEN("the gradient points to negative values") {
                REQUIRE(grad[0] == Approx(-18.0));
                REQUIRE(grad[1] == Approx(-4.0));
            }
        }
    }
}
