// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#ifndef MINIMIZE_BOOTSTRAP_INCLUDED_HPP
#define MINIMIZE_BOOTSTRAP_INCLUDED_HPP

#include <functional>
#include <random>

#include "minimize/detail/bootstrap.hpp"
#include "minimize/detail/vector_math.hpp"
#include "minimize/fit_results.hpp"
#include "minimize/measurement.hpp"

namespace minimize {

/** Callback type for the parameter minimization */
template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
using minimize_function_t = std::function<minimize::FitResults<NumberOfParameters>(
    const Function<InputDimensions, NumberOfParameters>&, const DataVector&, minimize::floating_t, std::size_t)>;

/** Bootstrap the error estimation by generating new measurements from the residuals.
 * Each of the new distributions is used to fit the function, then the standard
 * deviation for every parameter is used to estimate its error.
 *
 * The method to minimize the parameters can be given as callbacks.
 */
template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
minimize::FitResults<NumberOfParameters> bootstrap_errors(
    Function<InputDimensions, NumberOfParameters>& function, const DataVector& measurements,
    minimize_function_t<InputDimensions, NumberOfParameters, DataVector> minimizer,
    minimize::floating_t tolerance = 1e-15, std::size_t max_iterations = 16535) {
    auto results = minimizer(function, measurements, tolerance, max_iterations);
    function.set_parameters(results.optimized_values());

    std::vector<minimize::parameter_t<NumberOfParameters>> bootstrap_results;
    constexpr size_t num_steps = 16;
    bootstrap_results.reserve(num_steps);

    const auto residuals = minimize::detail::compute_residuals(function, measurements);
    for (size_t i = 0; i < num_steps; ++i) {
        const auto sample = minimize::detail::create_sample_data(function, measurements, residuals);
        const auto step_results = minimizer(function, sample, tolerance, max_iterations);
        bootstrap_results.push_back(step_results.optimized_values());
    }
    results.set_optimized_value_errors(minimize::detail::compute_stddev(bootstrap_results));

    return results;
}

} /* namespace minimize */

#endif /* MINIMIZE_BOOTSTRAP_INCLUDED_HPP */
