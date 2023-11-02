// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#ifndef MINIMIZE_WSSR_INCLUDED_HPP
#define MINIMIZE_WSSR_INCLUDED_HPP

#include "minimize/detail/vector_math.hpp"
#include "minimize/function.hpp"
#include "minimize/measurement.hpp"

namespace minimize {

/** Computes weighted sum of squared residuals with unity weights.  */
template <std::size_t InputDimensions, std::size_t NumberOfParameters>
minimize::floating_t compute_wssr(const Function<InputDimensions, NumberOfParameters>& fun,
                                  const MeasurementVector<InputDimensions>& vec,
                                  const parameter_t<NumberOfParameters>& par) {
    minimize::floating_t rv = 0.0;
    for (const auto& x : vec) {
        const auto diff = (fun.evaluate(x.in, par) - x.out);
        rv += diff * diff;
    }
    return rv;
}

/** Computes weighted sum of squared residuals with unity weights.  */
template <std::size_t InputDimensions, std::size_t NumberOfParameters>
minimize::floating_t compute_wssr(const Function<InputDimensions, NumberOfParameters>& fun,
                                  const MeasurementVector<InputDimensions>& vec) {
    return compute_wssr(fun, vec, fun.parameters());
}

/** Computes weighted sum of squared residuals with weights. */
template <std::size_t InputDimensions, std::size_t NumberOfParameters>
minimize::floating_t compute_wssr(const Function<InputDimensions, NumberOfParameters>& fun,
                                  const MeasurementVectorWithErrors<InputDimensions>& vec,
                                  const parameter_t<NumberOfParameters>& par) {
    minimize::floating_t rv = 0.0;
    for (const auto& x : vec) {
        const auto diff = (fun.evaluate(x.in, par) - x.out) / x.error;
        rv += (diff * diff);
    }
    return rv;
}

/** Computes weighted sum of squared residuals with weights. */
template <std::size_t InputDimensions, std::size_t NumberOfParameters>
minimize::floating_t compute_wssr(const Function<InputDimensions, NumberOfParameters>& fun,
                                  const MeasurementVectorWithErrors<InputDimensions>& vec) {
    return compute_wssr(fun, vec, fun.parameters());
}

/** Computes the gradient of wssr w.r.t. to the function parameters with unity weights.  */
template <std::size_t InputDimensions, std::size_t NumberOfParameters>
std::array<minimize::floating_t, NumberOfParameters> compute_wssr_gradient(
    const Function<InputDimensions, NumberOfParameters>& fun, const MeasurementVector<InputDimensions>& vec,
    const parameter_t<NumberOfParameters>& par) {
    std::array<minimize::floating_t, NumberOfParameters> rv;
    rv.fill(0.0);
    for (const auto& x : vec) {
        // sum 2*(f(x)-e) * f'(x)
        const auto factor = 2.0 * (fun.evaluate(x.in, par) - x.out);
        const auto grad = fun.parameter_gradient(x.in, par);
        detail::add_to_vector(rv, factor, grad);
    }
    return rv;
}

template <std::size_t InputDimensions, std::size_t NumberOfParameters>
std::array<minimize::floating_t, NumberOfParameters> compute_wssr_gradient(
    const Function<InputDimensions, NumberOfParameters>& fun, const MeasurementVector<InputDimensions>& vec) {
    return compute_wssr_gradient(fun, vec, fun.parameters());
}

/** Computes the gradient of wssr w.r.t. to the function parameters with weights.  */
template <std::size_t InputDimensions, std::size_t NumberOfParameters>
std::array<minimize::floating_t, NumberOfParameters> compute_wssr_gradient(
    const Function<InputDimensions, NumberOfParameters>& fun, const MeasurementVectorWithErrors<InputDimensions>& vec,
    const parameter_t<NumberOfParameters>& par) {
    std::array<minimize::floating_t, NumberOfParameters> rv;
    rv.fill(0.0);
    for (const auto& x : vec) {
        // wssr: sum ((f(x,p)-e)**2)/w
        // d/dp sum ((f(x,p)-e)**2)/w
        // sum 2*((f(x,p)-e)* f'(x,p))/w
        const auto factor = 2.0 * (fun.evaluate(x.in, par) - x.out) / x.error;
        const auto grad = fun.parameter_gradient(x.in, par);
        detail::add_to_vector(rv, factor, grad);
    }
    return rv;
}

template <std::size_t InputDimensions, std::size_t NumberOfParameters>
std::array<minimize::floating_t, NumberOfParameters> compute_wssr_gradient(
    const Function<InputDimensions, NumberOfParameters>& fun, const MeasurementVectorWithErrors<InputDimensions>& vec) {
    return compute_wssr_gradient(fun, vec, fun.parameters());
}

}  // namespace minimize

#endif /* MINIMIZE_WSSR_INCLUDED_HPP */
