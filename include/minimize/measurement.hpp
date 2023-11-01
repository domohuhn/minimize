// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#ifndef MINIMIZE_MEASUREMENT_INCLUDED_HPP
#define MINIMIZE_MEASUREMENT_INCLUDED_HPP

#include <array>
#include <vector>

#include "minimize/detail/meta.hpp"

namespace minimize {

/** Represents a measured data point. Used to fit the function parameters to match the measured values. */
template <std::size_t InputDimensions>
struct MeasurementWithError {
    using input_t = typename ::minimize::detail::type_selection_helper<InputDimensions>::type;
    input_t in{};
    floating_t out{0.0};
    floating_t error{0.0};
};

/** Represents a measured data point. Used to fit the function parameters to match the measured values. */
template <std::size_t InputDimensions>
struct Measurement {
    using input_t = typename ::minimize::detail::type_selection_helper<InputDimensions>::type;
    input_t in{};
    floating_t out{0.0};
};

template <std::size_t InputDimensions>
using MeasurementVector = std::vector<Measurement<InputDimensions>>;

template <std::size_t InputDimensions>
using MeasurementVectorWithErrors = std::vector<MeasurementWithError<InputDimensions>>;

}  // namespace minimize

#endif /* MINIMIZE_MEASUREMENT_INCLUDED_HPP */
