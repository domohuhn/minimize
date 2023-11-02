// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib
//
// A simple example showing how to use the library to fit a linear function (or any polynomial)
// to a set of measurement values.

#include <array>
#include <cstdint>
#include <iostream>
#include <random>

#include "minimize/minimize.hpp"

static std::array<double, 2> expected_parameters{-3.0, 4.0};

minimize::MeasurementVector<1> read_measurement_data() {
    minimize::MeasurementVector<1> data{};
    using m_t = minimize::Measurement<1>;
    // we generate a set of fake measurement values.
    // In this case: a linear function with gaussian noise.
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> gauss{0.0, 1.0};
    std::cout << "# Creating measurement data\n# x y\n";
    for (size_t i = 0; i < 101; ++i) {
        const double x = i;
        const double y = expected_parameters[0] + expected_parameters[1] * x + gauss(gen);
        std::cout << x << " " << y << "\n";
        data.emplace_back(m_t{x, y});
    }
    std::cout << "\n\n";
    return data;
}

int main() {
    const auto data = read_measurement_data();
    // create a polynomial of degree 1
    minimize::Polynomial<1> poly{};
    // set initial values
    poly.set_parameter(0, 0.0);
    poly.set_parameter(1, 0.0);
    // fit
    const auto results = minimize::conjugate_gradient_descent(poly, data);

    // print results

    std::cout << "# Fitting a linear function to measurement data with random noise.\n";
    std::cout << results.create_report();
    std::cout << "# parameters:\n";
    for (std::size_t i = 0; i < poly.number_of_parameters; ++i) {
        const auto diff = poly.parameter(i) - expected_parameters[i];
        std::cout << "# " << poly.parameter_name(i) << " : " << poly.parameter(i)
                  << "   (difference to real value: " << diff << ")\n";
    }

    return 0;
}
