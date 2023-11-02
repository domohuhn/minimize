// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#ifndef MINIMIZE_CONJUGATE_GRADIENT_DESCENT_INCLUDED_HPP
#define MINIMIZE_CONJUGATE_GRADIENT_DESCENT_INCLUDED_HPP

#include "minimize/find_minimum_on_line.hpp"
#include "minimize/fit_results.hpp"
#include "minimize/function.hpp"
#include "minimize/measurement.hpp"
#include "minimize/wssr.hpp"

namespace minimize {

namespace detail {

template <std::size_t NumberOfParameters>
minimize::floating_t compute_gamma(const minimize::parameter_t<NumberOfParameters>& gi,
                                   const minimize::parameter_t<NumberOfParameters>& gi_plus1) {
    minimize::floating_t denom = 0.0;
    minimize::floating_t num = 0.0;
    for (size_t i = 0; i < NumberOfParameters; ++i) {
        denom += gi[i] * gi[i];
        num += (gi_plus1[i] - gi[i]) * gi_plus1[i];
    }
    return num / denom;
}

template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
minimize::floating_t conjugate_gradient_descent_step(const Function<InputDimensions, NumberOfParameters>& function,
                                                     minimize::parameter_t<NumberOfParameters>& minimum,
                                                     const DataVector& measurements) {
    auto wssr = compute_wssr(function, measurements, minimum);
    auto gi = compute_wssr_gradient(function, measurements, minimum);
    auto conjugate_gradient = gi;
    for (size_t i = 0; i < NumberOfParameters; ++i) {
        const auto next_parameters = find_minimum_on_line(function, minimum, measurements, conjugate_gradient, 128);
        const auto next_wssr = compute_wssr(function, measurements, next_parameters);
        if (next_wssr >= wssr) {
            break;
        }
        wssr = next_wssr;
        minimum = next_parameters;
        const auto gi_plus1 = compute_wssr_gradient(function, measurements, minimum);
        const auto gamma = compute_gamma(gi, gi_plus1);
        gi = gi_plus1;
        conjugate_gradient = detail::axpy(gamma, conjugate_gradient, gi_plus1);
    }

    return wssr;
}

}  // namespace detail

template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
minimize::FitResults<NumberOfParameters> conjugate_gradient_descent(
    const Function<InputDimensions, NumberOfParameters>& function, const DataVector& measurements,
    minimize::floating_t tolerance = 1e-15, std::size_t max_iterations = 16535) {
    std::size_t iterations = 0;

    auto wssr = compute_wssr(function, measurements);
    minimize::FitResults<NumberOfParameters> results(wssr, measurements.size());
    results.initialize_before_fit(function);
    auto minimum = function.parameters();
    minimize::floating_t rel_change;
    do {
        const auto next_wssr = detail::conjugate_gradient_descent_step(function, minimum, measurements);
        if (next_wssr >= wssr || wssr == 0.0) {
            break;
        }
        rel_change = 1.0 - next_wssr / wssr;
        wssr = next_wssr;
        ++iterations;
    } while (iterations < max_iterations && tolerance < rel_change);
    results.set_converged(iterations < max_iterations);
    results.set_iterations(iterations);
    results.set_weighted_sum_of_squared_residuals(wssr);
    results.set_optimized_parameters(minimum);
    return results;
}

}  // namespace minimize

#endif /* MINIMIZE_CONJUGATE_GRADIENT_DESCENT_INCLUDED_HPP */
