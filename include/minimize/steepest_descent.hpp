// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#ifndef MINIMIZE_STEEPEST_DESCENT_INCLUDED_HPP
#define MINIMIZE_STEEPEST_DESCENT_INCLUDED_HPP

#include "minimize/bootstrap.hpp"
#include "minimize/find_minimum_on_line.hpp"
#include "minimize/function.hpp"
#include "minimize/measurement.hpp"
#include "minimize/wssr.hpp"

namespace minimize {

namespace detail {

template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
minimize::FitResults<NumberOfParameters> steepest_descent_impl(
    const Function<InputDimensions, NumberOfParameters>& function, const DataVector& measurements,
    minimize::floating_t tolerance = 1e-15, std::size_t max_iterations = 16535) {
    auto minimum = function.parameters();
    std::size_t iterations = 0;

    auto wssr = compute_wssr(function, measurements);

    minimize::FitResults<NumberOfParameters> results(wssr, measurements.size());
    results.initialize_before_fit(function);
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

    results.set_converged(iterations < max_iterations);
    results.set_iterations(iterations);
    results.set_weighted_sum_of_squared_residuals(wssr);
    results.set_optimized_values(minimum);
    return results;
}

}  // namespace detail

template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
minimize::FitResults<NumberOfParameters> steepest_descent(Function<InputDimensions, NumberOfParameters>& function,
                                                          const DataVector& measurements,
                                                          minimize::floating_t tolerance = 1e-15,
                                                          std::size_t max_iterations = 16535) {
    return minimize::bootstrap_errors<InputDimensions, NumberOfParameters, DataVector>(
        function, measurements,
        minimize::detail::steepest_descent_impl<InputDimensions, NumberOfParameters, DataVector>, tolerance,
        max_iterations);
}

}  // namespace minimize

#endif /* MINIMIZE_STEEPEST_DESCENT_INCLUDED_HPP */
