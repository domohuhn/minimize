// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#ifndef MINIMIZE_BOOTSTRAP_INCLUDED_HPP
#define MINIMIZE_BOOTSTRAP_INCLUDED_HPP

#include <functional>
#include <random>

#include "minimize/detail/vector_math.hpp"
#include "minimize/fit_results.hpp"
#include "minimize/measurement.hpp"

namespace minimize {

namespace detail {

/** Computes residuals between the function and the measured data. */
template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
std::vector<floating_t> compute_residuals(Function<InputDimensions, NumberOfParameters>& fun, const DataVector& vec) {
    std::vector<floating_t> rv;
    rv.reserve(vec.size());
    for (const auto& x : vec) {
        const auto diff = (fun.evaluate(x.in) - x.out);
        rv.push_back(diff);
    }
    return rv;
}

/** @brief Creates fake measurements by computing the value of the function and adding a randomly selected residual.
 *
 * Using the assumption that the residuals are all independent and sampled from the same distribution, creating data
 * this way gives an approximation of another measurement with the same random noise.
 */
template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
minimize::MeasurementVector<InputDimensions> create_sample_data(Function<InputDimensions, NumberOfParameters>& fun,
                                                                const DataVector& vec,
                                                                const std::vector<floating_t>& residuals) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<std::size_t> dist(0, vec.size());
    minimize::MeasurementVector<InputDimensions> rv;
    rv.reserve(vec.size());
    for (const auto& x : vec) {
        const auto y = fun.evaluate(x.in) + residuals[dist(gen)];
        rv.push_back(minimize::Measurement<InputDimensions>{x.in, y});
    }
    return rv;
}

/** compute mean */
template <std::size_t NumberOfParameters>
minimize::parameter_t<NumberOfParameters> compute_mean(
    const std::vector<minimize::parameter_t<NumberOfParameters>>& values) {
    minimize::parameter_t<NumberOfParameters> mean;
    mean.fill(0.0);
    for (const auto& v : values) {
        detail::add_to_vector(mean, 1.0, v);
    }
    return detail::scale_vector(floating_t(1.0 / values.size()), mean);
}

/** compute std dev */
template <std::size_t NumberOfParameters>
minimize::parameter_t<NumberOfParameters> compute_stddev(
    const std::vector<minimize::parameter_t<NumberOfParameters>>& values) {
    const auto mean = compute_mean(values);
    minimize::parameter_t<NumberOfParameters> variance;
    variance.fill(0.0);
    for (const auto& v : values) {
        detail::add_to_vector(variance, 1.0, detail::elementwise_a_minus_b_squared(v, mean));
    }
    variance = detail::scale_vector(floating_t(1.0 / values.size()), variance);
    return elementwise_sqrt(variance);
}

}  // namespace detail

/** bootstrap */

template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
using minimize_function_t = std::function<minimize::FitResults<NumberOfParameters>(
    const Function<InputDimensions, NumberOfParameters>&, const DataVector&, minimize::floating_t, std::size_t)>;

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
