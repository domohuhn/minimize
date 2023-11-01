// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#include <cmath>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common.hpp"
#include "minimize/function.hpp"

using Catch::Approx;

SCENARIO("Gradients can be numerically computed", "[function]") {
    GIVEN("A linear function") {
        LinearFunction linear{};
        WHEN("the gradient is computed at different positions") {
            const std::array<double, 1> x0{0.0};
            const std::array<double, 1> x1{1.0};
            const std::array<double, 1> x2{2.0};
            // using larger exponents creates floating point errors
            const std::array<double, 1> x3{1.0e8};
            const std::array<double, 1> x4{-1.0e8};
            const auto g0 = linear.parameter_gradient(x0);
            const auto g1 = linear.parameter_gradient(x1);
            const auto g2 = linear.parameter_gradient(x2);
            const auto g3 = linear.parameter_gradient(x3);
            const auto g4 = linear.parameter_gradient(x4);
            THEN("the first value of the gradient scales linearly with the position") {
                REQUIRE(g0[0] == Approx(0.0));
                REQUIRE(g1[0] == Approx(1.0));
                REQUIRE(g2[0] == Approx(2.0));
                REQUIRE(g3[0] == Approx(1.0e8));
                REQUIRE(g4[0] == Approx(-1.0e8));
            }

            THEN("the second value of the gradient is constant") {
                REQUIRE(g0[1] == Approx(1.0));
                REQUIRE(g1[1] == Approx(1.0));
                REQUIRE(g2[1] == Approx(1.0));
                REQUIRE(g3[1] == Approx(1.0));
                REQUIRE(g4[1] == Approx(1.0));
            }
        }
    }
}

class TanFunction : public minimize::Function<1, 4> {
public:
    TanFunction() : minimize::Function<1, 4>({2.0, 42.0, 0.1, 5.0}) {}

    virtual output_t evaluate(const input_t& x, const parameter_t& parameters) const {
        return parameters[0] * std::tan((x[0] - parameters[1]) * parameters[2]) + parameters[3];
    }
};

SCENARIO("Gradients of tangent", "[function]") {
    GIVEN("A tangent function") {
        TanFunction tan{};
        WHEN("the gradient is computed at 42") {
            const std::array<double, 1> x0{42.0};
            const auto g0 = tan.parameter_gradient(x0);
            THEN("the first result is ok") {
                REQUIRE(g0[0] == Approx(0.0));
                REQUIRE(g0[1] == Approx(-2.0 * 0.1));
                REQUIRE(g0[2] == Approx(0.0));
                REQUIRE(g0[3] == Approx(1.0));
            }
        }
    }
}

SCENARIO("Gradients of gaussian", "[function]") {
    GIVEN("A gaussian function") {
        Gaussian gauss{};
        WHEN("the gradient is computed at different positions") {
            const std::array<double, 1> x0{-5.0};
            const std::array<double, 1> x1{-3.0};
            const std::array<double, 1> x2{-8.0};
            const std::array<double, 1> x3{20.0};
            const std::array<double, 1> x4{-20.0};
            const auto g0 = gauss.parameter_gradient(x0);
            const auto g1 = gauss.parameter_gradient(x1);
            const auto g2 = gauss.parameter_gradient(x2);
            const auto g3 = gauss.parameter_gradient(x3);
            const auto g4 = gauss.parameter_gradient(x4);
            THEN("the first value is ok") {
                REQUIRE(g0[0] == Approx(-0.00000000000000780405));
                REQUIRE(g1[0] == Approx(0.0604927));
                REQUIRE(g2[0] == Approx(-0.0485691));
                REQUIRE(g3[0] == Approx(1.46725e-34));
                REQUIRE(g4[0] == Approx(-4.56435e-13));
            }

            THEN("the second value of the gradient is ok") {
                REQUIRE(g0[1] == Approx(-0.0997356));
                REQUIRE(g1[1] == Approx(-0.00000000000008454391));
                REQUIRE(g2[1] == Approx(0.0404742));
                REQUIRE(g3[1] == Approx(1.82232e-33));
                REQUIRE(g4[1] == Approx(3.36241e-12));
            }
        }
    }
}
