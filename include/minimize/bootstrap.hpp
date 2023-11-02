// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#ifndef MINIMIZE_BOOTSTRAP_INCLUDED_HPP
#define MINIMIZE_BOOTSTRAP_INCLUDED_HPP

#include <random>

#include "minimize/fit_results.hpp"
#include "minimize/measurement.hpp"

namespace minimize {

namespace detail {

/** Computes residuals between the function and the measured data. */
template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
std::vector<floating_t> compute_residuals(Function<InputDimensions, NumberOfParameters>& fun,
                                          const MeasurementVector<InputDimensions>& vec) {
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
                                                                const MeasurementVector<InputDimensions>& vec,
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

/** compute std dev */

}  // namespace detail

/** bootstrap */

} /* namespace minimize */

#endif /* MINIMIZE_BOOTSTRAP_INCLUDED_HPP */
