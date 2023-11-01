// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#include <cmath>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "common.hpp"
#include "minimize/conjugate_gradient_descent.hpp"

using Catch::Approx;
using namespace minimize;

SCENARIO("Conjugate gradient descent: fit linear function", "[conjugate gradient]") {
    GIVEN("Perfect measurement data") {
        LinearFunction linear{};
        MeasurementVector<1> vec{};
        for (size_t i = 0; i < 100; ++i) {
            using in_t = minimize::floating_t;
            using m_t = minimize::Measurement<1>;
            vec.emplace_back(m_t{in_t{0.25 * i}, 4.0 * i - 3.0});
        }

        WHEN("the minimum is searched") {
            const auto chi2 = conjugate_gradient_descent(linear, vec, 1.0e-15);
            const auto found = linear.parameters();
            THEN("a minimum is found") {
                REQUIRE_THAT(chi2, Catch::Matchers::WithinAbs(0.0, 1e-18));
                REQUIRE_THAT(found[0], Catch::Matchers::WithinRel(16.0, 1e-14));
                REQUIRE_THAT(found[1], Catch::Matchers::WithinRel(-3.0, 1e-12));
            }
        }
    }

    GIVEN("Noisy measurement data") {
        LinearFunction linear{};
        MeasurementVector<1> vec{};
        for (size_t i = 0; i < 100; ++i) {
            using in_t = minimize::floating_t;
            using m_t = minimize::Measurement<1>;
            vec.emplace_back(m_t{in_t{0.25 * i}, 1.5 * i + 27.9 + 0.1 * (i % 3)});
        }

        WHEN("the minimum is searched") {
            const auto chi2 = conjugate_gradient_descent(linear, vec, 1.0e-15);
            const auto found = linear.parameters();
            THEN("a minimum is found") {
                REQUIRE_THAT(found[0], Catch::Matchers::WithinRel(6.0, 1e-4));
                REQUIRE_THAT(found[1], Catch::Matchers::WithinRel(28.0, 1e-4));
                REQUIRE_THAT(chi2, Catch::Matchers::WithinAbs(0.67, 1e-2));
            }
        }
    }
}

SCENARIO("Conjugate gradient descent: gaussian function", "[conjugate gradient]") {
    GIVEN("Perfect measurement data") {
        Gaussian gauss{};
        MeasurementVector<1> vec{};
        for (size_t i = 0; i < 100; ++i) {
            using in_t = minimize::floating_t;
            using m_t = minimize::Measurement<1>;
            vec.emplace_back(m_t{in_t{0.25 * i}, compute_gaussian(0.25 * i)});
        }
        gauss.set_parameters(::minimize::parameter_t<2>{0.0, 1.0});

        WHEN("the minimum is searched") {
            const auto chi2 = conjugate_gradient_descent(gauss, vec, 1.0e-15);
            const auto found = gauss.parameters();
            THEN("a minimum is found") {
                REQUIRE_THAT(found[0], Catch::Matchers::WithinRel(14.0, 1e-8));
                REQUIRE_THAT(found[1], Catch::Matchers::WithinRel(2.5, 1e-8));
                REQUIRE_THAT(chi2, Catch::Matchers::WithinAbs(0.0, 1e-8));
            }
        }
    }
}

SCENARIO("Conjugate gradient descent: fit saddle function", "[conjugate gradient]") {
    GIVEN("Perfect measurement data") {
        SaddleFunction saddle{};
        // measurements for 0.5, 1.0, 1.3, 5.0
        MeasurementVector<2> vec = create_perfect_test_data_saddle();

        WHEN("the minimum is searched") {
            const auto chi2 = conjugate_gradient_descent(saddle, vec, 1.0e-9);
            THEN("a minimum is found") {
                // there are infinitely many solutions
                // => check that chi2 converged.
                REQUIRE_THAT(chi2, Catch::Matchers::WithinAbs(0.0, 1e-18));
            }
        }
    }
}
