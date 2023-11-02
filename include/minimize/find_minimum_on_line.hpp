// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#ifndef MINIMIZE_FIND_MINIMUM_ON_LINE_INCLUDED_HPP
#define MINIMIZE_FIND_MINIMUM_ON_LINE_INCLUDED_HPP

#include <utility>

#include "minimize/detail/vector_math.hpp"
#include "minimize/function.hpp"
#include "minimize/measurement.hpp"
#include "minimize/wssr.hpp"

namespace minimize {

template <std::size_t InputDimensions>
struct Interval {
    parameter_t<InputDimensions> before;
    parameter_t<InputDimensions> past;
};

/** This method searches a point that is just past the minimum in the parameter space of a function.
 *
 * @param[in] fun function to search the minimum
 * @param[in] par parameters for the function
 * @param[in] vec measured data points
 * @param[in] direction direction of the search in the parameter space
 * @param[in] max_iterations limit for the iterations in the search
 */
template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
Interval<NumberOfParameters> search_interval_around_minimum(const Function<InputDimensions, NumberOfParameters>& fun,
                                                            const parameter_t<NumberOfParameters>& par,
                                                            const DataVector& vec,
                                                            const parameter_t<NumberOfParameters>& direction,
                                                            std::size_t max_iterations) {
    std::size_t iterations = 0;
    parameter_t<NumberOfParameters> before = par;
    parameter_t<NumberOfParameters> mid = par;
    parameter_t<NumberOfParameters> past;
    minimize::floating_t last_wssr = compute_wssr(fun, vec, par);
    bool is_smaller = true;
    minimize::floating_t current_position = 0.01;
    const minimize::floating_t scale_factor = 1.618;
    do {
        past = detail::axpy(-current_position, direction, par);
        const minimize::floating_t next_wssr = compute_wssr(fun, vec, past);
        is_smaller = next_wssr < last_wssr;
        // We assume a single global minimum along this direction.
        // On step x we may step past the minimum, but the wssr is still smaller than
        // in the previous iteration. So returning p[i-1],p[i] as interval may not include
        // the minimum. Returning p[i-2],p[i] fixes this problem.
        if (is_smaller) {
            before = mid;
            mid = past;
        }
        last_wssr = next_wssr;
        current_position *= scale_factor;
        ++iterations;
    } while (iterations < max_iterations && is_smaller);

    return {before, past};
}

template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
Interval<NumberOfParameters> search_interval_around_minimum(const Function<InputDimensions, NumberOfParameters>& fun,
                                                            const DataVector& vec,
                                                            const parameter_t<NumberOfParameters>& direction,
                                                            std::size_t max_iterations) {
    return search_interval_around_minimum(fun, fun.parameters(), vec, direction, max_iterations);
}

/**
 * @brief Performs a binary search between lower and upper searching
 * the minimum.
 *
 * The method will compute the wssr in the middle between the two points,
 * then construct a parabola using the following three points:
 * (-1, wssr_lower), (0, wssr_mid), (1, chi_upper).
 *
 * The next interval is the one with the vertex inside.
 * These two steps are repeated until a minimum is found.
 *
 * @param fun Function to minimize
 * @param vec Measured data
 * @param lower lower bound of the parameter interval
 * @param upper upper bound of the parameter interval
 * @param max_iterations iteration limit
 * @return parameter_t<NumberOfParameters>
 */
template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
parameter_t<NumberOfParameters> binary_search_minimum_in_interval(
    const Function<InputDimensions, NumberOfParameters>& fun, const DataVector& vec,
    parameter_t<NumberOfParameters> lower, parameter_t<NumberOfParameters> upper, std::size_t max_iterations) {
    std::size_t iterations = 0;
    minimize::floating_t lower_wssr = compute_wssr(fun, vec, lower);
    minimize::floating_t upper_wssr = compute_wssr(fun, vec, upper);
    minimize::floating_t rel_change = 0.0;

    do {
        parameter_t<NumberOfParameters> mid = detail::lerp(0.5, lower, upper);
        const minimize::floating_t mid_wssr = compute_wssr(fun, vec, mid);
        if (mid_wssr == 0) {
            return mid;
        }

        const minimize::floating_t half = 0.5;
        const minimize::floating_t quarter = 0.25;
        const minimize::floating_t parabola_opening = (lower_wssr + upper_wssr) * half - mid_wssr;
        const minimize::floating_t parabola_vertex = quarter * ((lower_wssr - upper_wssr) / parabola_opening);

        if (parabola_opening > 0) {
            if (parabola_vertex < 0) {
                upper = mid;
                upper_wssr = mid_wssr;
            } else if (parabola_vertex > 0) {
                lower = mid;
                lower_wssr = mid_wssr;
            } else {
                // special case - only happens if lower_wssr==upper_wssr and mid_wssr<upper_wssr
                // reduce interval on both sides by an asymmetric amount
                const auto tmp = detail::lerp(0.01, lower, upper);
                const auto tmp2 = detail::lerp(0.98, lower, upper);
                lower = tmp;
                upper = tmp2;
                lower_wssr = compute_wssr(fun, vec, lower);
                upper_wssr = compute_wssr(fun, vec, upper);
            }
        } else {
            // special case - middle has worse wssr than outer positions
            // we are done.
            break;
        }
        ++iterations;
    } while (iterations < max_iterations);
    if (lower_wssr < upper_wssr) {
        return lower;
    } else {
        return upper;
    }
}

/**
 * @brief Finds a minimum of the wssr in the opposite direction of the gradient.
 *
 * @param[in] fun Function to minimize
 * @param[in] par parameters for the function
 * @param[in] vec Measured data
 * @param[in] gradient Gradient of wssr wrt to the function parameters
 * @param[in] max_iterations Max number of iterations
 * @return Found minimum
 */
template <std::size_t InputDimensions, std::size_t NumberOfParameters, typename DataVector>
typename Function<InputDimensions, NumberOfParameters>::parameter_t find_minimum_on_line(
    const Function<InputDimensions, NumberOfParameters>& fun, const parameter_t<NumberOfParameters>& par,
    const DataVector& vec, const parameter_t<NumberOfParameters>& gradient, std::size_t max_iterations) {
    const auto bracket = search_interval_around_minimum(fun, par, vec, gradient, max_iterations);
    return binary_search_minimum_in_interval(fun, vec, bracket.before, bracket.past, max_iterations);
}

}  // namespace minimize

#endif /* MINIMIZE_FIND_MINIMUM_ON_LINE_INCLUDED_HPP */
