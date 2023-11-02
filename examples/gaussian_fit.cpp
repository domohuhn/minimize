// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib
//
// A simple example showing how to use the library to fit a 2d - gaussian curve
// to a set of measurement values.

#include <array>
#include <cstdint>
#include <iostream>
#include <random>

#include "minimize/minimize.hpp"

static std::array<double, 6> expected_parameters{-3.0, 4.0, 1.3, 2.1, 160.2, 3.0};

/**
 * To use your own function, you must subclass
 * minimize::Function. The first template argument
 * is the number of input dimensions, here 2
 * as we want a 2-d gaussian.
 *
 * Each gaussian has 2 parameters: mean, stddev.
 * There is also a common amplitude and offset, so a total of 6.
 */
class Gaussian : public minimize::Function<2, 6> {
public:
    Gaussian() : minimize::Function<2, 6>(expected_parameters) {}

    /** Override this method to create a custom function.
     * x is a std::array with the size of the number of input dimensions (here 2).
     * For dimension 1 x is just a double.
     *
     * parameters is a std::array with the size of the number of parameters (here 6).
     */
    output_t evaluate(const input_t& x, const parameter_t& parameters) const override {
        return gaussian1d(x[0], parameters[0], parameters[1]) * gaussian1d(x[1], parameters[2], parameters[3]) *
                   parameters[4] +
               parameters[5];
    }
    using minimize::Function<2, 6>::evaluate;

    minimize::floating_t gaussian1d(minimize::floating_t x, minimize::floating_t mean,
                                    minimize::floating_t stddev) const {
        const auto arg = (x - mean) / stddev;
        return 1.0 / (stddev * std::sqrt(2 * 3.14159265358979323846264338327950288419716939937510)) *
               std::exp(-0.5 * arg * arg);
    }

    /** custom parameter names in the report - optional. */
    std::string parameter_name(size_t i) const override {
        switch (i) {
            case 0:
                return "mean x";
            case 1:
                return "stddev x";
            case 2:
                return "mean y";
            case 3:
                return "stddev y";
            case 4:
                return "amplitude";
            case 5:
                return "offset";
            default:
                return minimize::Function<2, 6>::parameter_name(i);
        }
    }
};

minimize::MeasurementVector<2> read_measurement_data(bool verbose) {
    minimize::MeasurementVector<2> data{};
    using m_t = minimize::Measurement<2>;
    using in_t = typename minimize::Measurement<2>::input_t;
    // we generate a set of fake measurement values.
    // we also add gaussian noise.
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> errors{0.0, 0.25};
    if (verbose) std::cout << "# Creating measurement data\n# x y z\n";
    Gaussian gauss{};
    for (int i = 0; i < 20; ++i) {
        for (int k = 0; k < 20; ++k) {
            const double x = i - 10;
            const double y = k - 10;
            in_t position{x, y};
            const double z = gauss.evaluate(position) + errors(gen);
            if (verbose) std::cout << x << " " << y << " " << z << "\n";
            data.emplace_back(m_t{position, z});
        }
    }
    if (verbose) std::cout << "\n\n";
    return data;
}

int main(int argc, char**) {
    if (argc <= 1) {
        std::cout << "Pass any argument to see the generated data points\n";
    }
    const auto data = read_measurement_data(argc > 1);
    // create a gaussian curve
    Gaussian gauss{};
    // set initial values
    gauss.set_parameter(0, 0.0);
    gauss.set_parameter(1, 1.0);
    gauss.set_parameter(2, 0.0);
    gauss.set_parameter(3, 1.0);
    gauss.set_parameter(4, 1.0);
    gauss.set_parameter(5, 0.0);
    // fit
    const auto results = minimize::conjugate_gradient_descent(gauss, data);

    // print results

    std::cout << "# Fitting a linear function to measurement data with random noise.\n";
    std::cout << results.create_report();
    std::cout << "# parameters:\n";
    for (std::size_t i = 0; i < gauss.number_of_parameters; ++i) {
        const auto diff = gauss.parameter(i) - expected_parameters[i];
        std::cout << "# " << gauss.parameter_name(i) << " : " << gauss.parameter(i)
                  << "   (difference to real value: " << diff << ")\n";
    }

    return 0;
}
