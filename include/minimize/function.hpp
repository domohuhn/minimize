// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#ifndef MINIMIZE_FUNCTION_INCLUDED_HPP
#define MINIMIZE_FUNCTION_INCLUDED_HPP

#include <array>
#include <cmath>
#include <cstdio>
#include <string>

#include "minimize/detail/meta.hpp"

namespace minimize {

template <std::size_t NumberOfParameters>
using parameter_t = std::array<floating_t, NumberOfParameters>;

template <std::size_t InputDimensions, std::size_t NumberOfParameters>
class Function {
public:
    using parameter_t = ::minimize::parameter_t<NumberOfParameters>;
    using input_t = typename ::minimize::detail::type_selection_helper<InputDimensions>::type;
    using output_t = floating_t;
    static constexpr std::size_t input_dimensions = InputDimensions;
    static constexpr std::size_t number_of_parameters = NumberOfParameters;

    Function() = default;
    Function(const parameter_t& p) : parameters_(p) {}

    virtual ~Function() {}

    /** Returns the name of the i-th parameter. Used to create the report after fitting.
     * You can override this method to return names other than the default name.
     */
    virtual std::string parameter_name(size_t i) const { return "p" + std::to_string(i); }

    virtual output_t evaluate(const input_t& x, const parameter_t& parameters) const = 0;

    output_t evaluate(const input_t& x) const { return evaluate(x, parameters_); }

    /**
     * @brief Computes the gradient wrt. to the parameters.
     *
     * This method numerically computes the gradient using a five point stencil.
     *
     * @param x the current position
     * @param parameters the current parameters to use.
     * @return parameter_t the computed gradient
     */
    virtual parameter_t parameter_gradient(const input_t& x, const parameter_t& parameters) const {
        parameter_t gradient;
        const floating_t dh = std::sqrt(std::sqrt(numerical_differentiation_epsilon()));
        for (std::size_t i = 0; i < gradient.size(); ++i) {
            parameter_t copy = parameters;
            // 5 point stencil
            const floating_t dp = parameters[i] != 0.0 ? parameters[i] * dh : dh;
            volatile floating_t p_plus1 = parameters[i] + dp;
            volatile floating_t p_plus2 = parameters[i] + 2.0 * dp;
            volatile floating_t p_minus1 = parameters[i] - dp;
            volatile floating_t p_minus2 = parameters[i] - 2.0 * dp;
            const floating_t dx = 3.0 * (p_plus2 - p_minus2);

            copy[i] = p_plus2;
            const floating_t f_plus2 = evaluate(x, copy);
            copy[i] = p_plus1;
            const floating_t f_plus1 = evaluate(x, copy);
            copy[i] = p_minus1;
            const floating_t f_minus1 = evaluate(x, copy);
            copy[i] = p_minus2;
            const floating_t f_minus2 = evaluate(x, copy);

            gradient[i] = (-f_plus2 + 8.0 * f_plus1 - 8.0 * f_minus1 + f_minus2) / dx;
        }
        return gradient;
    }

    parameter_t parameter_gradient(const input_t& x) const { return parameter_gradient(x, parameters_); }

    void set_parameters(const parameter_t& p) { parameters_ = p; }

    void set_parameter(std::size_t i, floating_t p) { parameters_[i] = p; }

    const parameter_t& parameters() const { return parameters_; }

    floating_t parameter(std::size_t i) const { return parameters_[i]; }

    void set_numerical_differentiation_epsilon(floating_t p) noexcept { epsilon_ = p; }

    floating_t numerical_differentiation_epsilon() const noexcept { return epsilon_; }

private:
    parameter_t parameters_{};
    floating_t epsilon_{1e-15};
};

/** Linear functions */
template <std::size_t Degree>
class Polynomial : public minimize::Function<1, Degree + 1> {
public:
    using base_t = typename minimize::Function<1, Degree + 1>;
    using minimize::Function<1, Degree + 1>::Function;
    using output_t = typename base_t::output_t;
    using input_t = typename base_t::input_t;
    using parameter_t = typename base_t::parameter_t;

    virtual floating_t evaluate(const input_t& x, const parameter_t& parameters) const {
        minimize::floating_t t = 1.0;
        minimize::floating_t rv = parameters[0];
        for (size_t i = 1; i <= Degree; ++i) {
            t *= x;
            rv += t * parameters[i];
        }
        return rv;
    }

    using base_t::evaluate;
};

}  // namespace minimize

#endif /* MINIMIZE_FUNCTION_INCLUDED_HPP */
