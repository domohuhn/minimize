// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#ifndef MINIMIZE_FIT_RESULTS_INCLUDED_HPP
#define MINIMIZE_FIT_RESULTS_INCLUDED_HPP

#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

#include "minimize/detail/meta.hpp"
#include "minimize/function.hpp"

namespace minimize {

template <std::size_t NumberOfParameters>
class FitResults {
public:
    using parameter_t = ::minimize::parameter_t<NumberOfParameters>;
    static constexpr std::size_t number_of_parameters = NumberOfParameters;

    FitResults(floating_t wssr, std::size_t number_of_data_points)
        : initial_weighted_sum_of_squared_residuals_(wssr), number_of_data_points_(number_of_data_points) {}

    std::size_t degrees_of_freedom() const noexcept { return number_of_data_points_ - number_of_parameters; }

    floating_t weighted_sum_of_squared_residuals() const noexcept { return weighted_sum_of_squared_residuals_; }

    floating_t initial_weighted_sum_of_squared_residuals() const noexcept {
        return initial_weighted_sum_of_squared_residuals_;
    }

    floating_t normalized_weighted_sum_of_squared_residuals() const noexcept {
        return weighted_sum_of_squared_residuals_ / degrees_of_freedom();
    }

    const parameter_t& initial_values() const noexcept { return initial_values_; }

    const parameter_t& optimized_values() const noexcept { return optimized_parameters_; }

    const parameter_t& optimized_value_errors() const noexcept { return optimized_parameter_errors_; }

    void set_optimized_values(const parameter_t& p) { optimized_parameters_ = p; }

    void set_optimized_value_errors(const parameter_t& p) { optimized_parameter_errors_ = p; }

    void set_weighted_sum_of_squared_residuals(const floating_t& p) { weighted_sum_of_squared_residuals_ = p; }

    void set_converged(bool v) noexcept { converged_ = v; }

    bool converged() const noexcept { return converged_; }

    std::string create_report() const {
        std::stringstream stream;
        stream << "Fit Results\n";
        stream << "Data points        : " << number_of_data_points_ << "\n";
        stream << "Parameters         : " << NumberOfParameters << "\n";
        stream << "Degrees of freedom : " << degrees_of_freedom() << "\n";
        stream << "Initial WSSR       : " << initial_weighted_sum_of_squared_residuals_ << "\n";
        stream << "\n";
        stream << "Initial set of parameters:\n";

        for (std::size_t i = 0; i < NumberOfParameters; ++i) {
            stream << std::setw(20) << parameter_names[i] << " : " << initial_values_[i] << "\n";
        }
        stream << "\n\n";

        stream << "Iterations   : " << iterations_ << "\n";
        stream << "Converged    : " << std::boolalpha << converged_ << "\n";
        stream << "WSSR         : " << weighted_sum_of_squared_residuals() << "\n";
        stream << "WSSR/NDF     : " << normalized_weighted_sum_of_squared_residuals() << "\n";
        stream << "\n";
        stream << "Final set of parameters:\n";

        stream << std::setw(20) << "name"
               << " | " << std::setw(20) << "value"
               << " +- "
               << "error"
               << "\n";

        const auto default_precision{stream.precision()};
        for (std::size_t i = 0; i < NumberOfParameters; ++i) {
            stream << std::setw(20) << parameter_names[i] << " | " << std::setw(20) << optimized_parameters_[i]
                   << " +- " << optimized_parameter_errors_[i] << " (" << std::setprecision(2)
                   << std::abs(100.0 * optimized_parameter_errors_[i] / optimized_parameters_[i]) << " %)\n";
            stream << std::setprecision(default_precision);
        }
        stream << "\n\n";

        return stream.str();
    }

    template <std::size_t dim, std::size_t pars>
    void initialize_before_fit(const Function<dim, pars>& f) {
        static_assert(pars == NumberOfParameters, "The number of parameters must be identical!");
        initial_values_ = f.parameters();
        for (std::size_t i = 0; i < NumberOfParameters; ++i) {
            parameter_names[i] = f.parameter_name(i);
        }
    }

    void set_iterations(std::size_t s) { iterations_ = s; }

private:
    parameter_t initial_values_{};
    floating_t initial_weighted_sum_of_squared_residuals_{};
    std::size_t number_of_data_points_{0};
    std::size_t iterations_{0};
    bool converged_{false};
    floating_t weighted_sum_of_squared_residuals_{};
    std::array<std::string, NumberOfParameters> parameter_names{};

    parameter_t optimized_parameters_{};
    parameter_t optimized_parameter_errors_{};
};

} /* namespace minimize */

#endif /* MINIMIZE_FIT_RESULTS_INCLUDED_HPP */
