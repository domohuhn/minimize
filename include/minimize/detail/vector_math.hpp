// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#ifndef MINIMIZE_DETAIL_VECTOR_MATH_INCLUDED_HPP
#define MINIMIZE_DETAIL_VECTOR_MATH_INCLUDED_HPP

#include "minimize/function.hpp"
#include "minimize/measurement.hpp"

namespace minimize {

namespace detail {

template <std::size_t NumberOfParameters>
void add_to_vector(std::array<minimize::floating_t, NumberOfParameters>& out, minimize::floating_t scale,
                   const std::array<minimize::floating_t, NumberOfParameters>& in) {
    for (size_t i = 0; i < NumberOfParameters; ++i) {
        out[i] += scale * in[i];
    }
}

/** Computes "alpha times x plus y"
 */
template <std::size_t NumberOfParameters>
std::array<minimize::floating_t, NumberOfParameters> axpy(
    minimize::floating_t alpha, const std::array<minimize::floating_t, NumberOfParameters>& x,
    const std::array<minimize::floating_t, NumberOfParameters>& y) {
    std::array<minimize::floating_t, NumberOfParameters> rv;
    for (size_t i = 0; i < NumberOfParameters; ++i) {
        rv[i] = alpha * x[i] + y[i];
    }
    return rv;
}

/** elementwise computation of (a-b)*(a-b) */
template <std::size_t NumberOfParameters>
std::array<minimize::floating_t, NumberOfParameters> elementwise_a_minus_b_squared(
    const std::array<minimize::floating_t, NumberOfParameters>& a,
    const std::array<minimize::floating_t, NumberOfParameters>& b) {
    std::array<minimize::floating_t, NumberOfParameters> rv;
    for (size_t i = 0; i < NumberOfParameters; ++i) {
        rv[i] = (a[i] - b[i]) * (a[i] - b[i]);
    }
    return rv;
}

/** elementwise computation of sqrt */
template <std::size_t NumberOfParameters>
std::array<minimize::floating_t, NumberOfParameters> elementwise_sqrt(
    const std::array<minimize::floating_t, NumberOfParameters>& a) {
    std::array<minimize::floating_t, NumberOfParameters> rv;
    for (size_t i = 0; i < NumberOfParameters; ++i) {
        rv[i] = std::sqrt(a[i]);
    }
    return rv;
}

template <std::size_t NumberOfParameters>
std::array<minimize::floating_t, NumberOfParameters> lerp(
    minimize::floating_t alpha, const std::array<minimize::floating_t, NumberOfParameters>& x,
    const std::array<minimize::floating_t, NumberOfParameters>& y) {
    std::array<minimize::floating_t, NumberOfParameters> rv;
    for (size_t i = 0; i < NumberOfParameters; ++i) {
        rv[i] = (1.0 - alpha) * x[i] + alpha * y[i];
    }
    return rv;
}

template <std::size_t NumberOfParameters>
std::array<minimize::floating_t, NumberOfParameters> scale_vector(
    minimize::floating_t alpha, const std::array<minimize::floating_t, NumberOfParameters>& x) {
    std::array<minimize::floating_t, NumberOfParameters> rv;
    for (size_t i = 0; i < NumberOfParameters; ++i) {
        rv[i] = alpha * x[i];
    }
    return rv;
}

}  // namespace detail
}  // namespace minimize

#endif /* MINIMIZE_DETAIL_VECTOR_MATH_INCLUDED_HPP */
