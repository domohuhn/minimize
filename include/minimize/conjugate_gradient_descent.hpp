// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#ifndef MINIMIZE_STEEPEST_DESCENT_INCLUDED_HPP
#define MINIMIZE_STEEPEST_DESCENT_INCLUDED_HPP

#include "minimize/chi2.hpp"
#include "minimize/find_minimum_on_line.hpp"
#include "minimize/function.hpp"
#include "minimize/measurement.hpp"

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
minimize::floating_t conjugate_gradient_descent_step(Function<InputDimensions, NumberOfParameters>& function,
                                                     const DataVector& measurements) {
    auto chi2 = compute_chi2(function, measurements);
    auto gi = compute_chi2_gradient(function, measurements);
    auto conjugate_gradient = gi;
    for (size_t i = 0; i < NumberOfParameters; ++i) {
        const auto next_parameters = find_minimum_on_line(function, measurements, conjugate_gradient, 128);
        const auto next_chi2 = compute_chi2(function, measurements, next_parameters);
        if (next_chi2 >= chi2) {
            break;
        }
        chi2 = next_chi2;
        function.set_parameters(next_parameters);
        auto gi_plus1 = compute_chi2_gradient(function, measurements);
        const auto gamma = compute_gamma(gi, gi_plus1);
        gi = gi_plus1;
        conjugate_gradient = detail::axpy(gamma, conjugate_gradient, gi_plus1);
    }

    return chi2;
}

}  // namespace detail

template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
minimize::floating_t conjugate_gradient_descent(Function<InputDimensions, NumberOfParameters>& function,
                                                const DataVector& measurements, minimize::floating_t tolerance,
                                                std::size_t max_iterations = 16535) {
    std::size_t iterations = 0;

    auto chi2 = compute_chi2(function, measurements);
    minimize::floating_t rel_change;
    do {
        const auto next_chi2 = detail::conjugate_gradient_descent_step(function, measurements);
        if (next_chi2 >= chi2 || chi2 == 0.0) {
            break;
        }
        rel_change = 1.0 - next_chi2 / chi2;
        chi2 = next_chi2;
        ++iterations;
    } while (iterations < max_iterations && tolerance < rel_change);
    return chi2;
}

}  // namespace minimize

#endif /* MINIMIZE_STEEPEST_DESCENT_INCLUDED_HPP */
