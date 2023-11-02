// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#ifndef MINIMIZE_STEEPEST_DESCENT_INCLUDED_HPP
#define MINIMIZE_STEEPEST_DESCENT_INCLUDED_HPP

#include "minimize/find_minimum_on_line.hpp"
#include "minimize/function.hpp"
#include "minimize/measurement.hpp"
#include "minimize/wssr.hpp"

namespace minimize {

template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
minimize::floating_t steepest_descent(Function<InputDimensions, NumberOfParameters>& function,
                                      const DataVector& measurements, minimize::floating_t tolerance = 1e-15,
                                      std::size_t max_iterations = 16535) {
    auto minimum = function.parameters();
    std::size_t iterations = 0;

    auto wssr = compute_wssr(function, measurements);
    auto rel_change = 10.0 * tolerance;
    do {
        auto gradient = compute_wssr_gradient(function, measurements, minimum);
        const auto next_parameters = find_minimum_on_line(function, minimum, measurements, gradient, 128);
        const auto next_wssr = compute_wssr(function, measurements, next_parameters);
        if (next_wssr >= wssr || wssr == 0.0) {
            break;
        }
        minimum = next_parameters;
        rel_change = 1.0 - next_wssr / wssr;
        wssr = next_wssr;

        ++iterations;
    } while (iterations < max_iterations && tolerance < rel_change);

    function.set_parameters(minimum);
    return wssr;
}

}  // namespace minimize

#endif /* MINIMIZE_STEEPEST_DESCENT_INCLUDED_HPP */
