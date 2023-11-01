// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#ifndef MINIMIZE_STEEPEST_DESCENT_INCLUDED_HPP
#define MINIMIZE_STEEPEST_DESCENT_INCLUDED_HPP

#include "minimize/chi2.hpp"
#include "minimize/find_minimum_on_line.hpp"
#include "minimize/function.hpp"
#include "minimize/measurement.hpp"

namespace minimize {

template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
typename Function<InputDimensions, NumberOfParameters>::parameter_t steepest_descent(
    Function<InputDimensions, NumberOfParameters>& function, const DataVector& measurements,
    minimize::floating_t tolerance, std::size_t max_iterations = 16535) {
    auto minimum = function.parameters();
    std::size_t iterations = 0;

    auto chi2 = compute_chi2(function, measurements);
    auto rel_change = 10.0 * tolerance;
    do {
        auto gradient = compute_chi2_gradient(function, measurements);
        const auto next_parameters = find_minimum_on_line(function, measurements, gradient, 128);
        const auto next_chi2 = compute_chi2(function, measurements, next_parameters);
        if (next_chi2 >= chi2) {
            break;
        }
        minimum = next_parameters;
        rel_change = 1.0 - next_chi2 / chi2;
        chi2 = next_chi2;
        function.set_parameters(next_parameters);

        ++iterations;
    } while (iterations < max_iterations && tolerance < rel_change);

    return minimum;
}

}  // namespace minimize

#endif /* MINIMIZE_STEEPEST_DESCENT_INCLUDED_HPP */
