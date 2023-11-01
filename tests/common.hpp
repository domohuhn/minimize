// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#ifndef MINIMIZE_COMMON_INCLUDED_HPP
#define MINIMIZE_COMMON_INCLUDED_HPP

#include "minimize/function.hpp"
#include "minimize/measurement.hpp"

class LinearFunction : public minimize::Function<1, 2> {
public:
    LinearFunction() : minimize::Function<1, 2>({2.0, 42.0}) {}

    virtual output_t evaluate(const input_t& x, const parameter_t& parameters) const {
        return parameters[0] * x + parameters[1];
    }

    using minimize::Function<1, 2>::evaluate;
};

class Gaussian : public minimize::Function<1, 2> {
public:
    Gaussian() : minimize::Function<1, 2>({-5, 2.0}) {}

    virtual output_t evaluate(const input_t& x, const parameter_t& parameters) const {
        const auto arg = (x - parameters[0]) / parameters[1];
        return 1.0 / (parameters[1] * std::sqrt(2 * 3.14159265358979323846264338327950288419716939937510)) *
               std::exp(-0.5 * arg * arg);
    }
};

inline double compute_gaussian(double x) {
    double mean = 14;
    double stddev = 2.5;
    const auto arg = (x - mean) / stddev;
    return 1.0 / (stddev * std::sqrt(2 * 3.14159265358979323846264338327950288419716939937510)) *
           std::exp(-0.5 * arg * arg);
}

class SaddleFunction : public minimize::Function<2, 4> {
public:
    SaddleFunction() : minimize::Function<2, 4>({0.25, 0.5, 0.65, 2.5}) {}

    virtual output_t evaluate(const input_t& x, const parameter_t& parameters) const {
        return parameters[0] * x[0] * x[0] + parameters[1] * x[1] * x[1] + parameters[2] * x[0] * x[1] + parameters[3];
    }

    using minimize::Function<2, 4>::evaluate;
};

inline minimize::MeasurementVector<2> create_perfect_test_data_saddle() {
    // measurements for 0.5, 1.0, 1.3, 5.0
    minimize::MeasurementVector<2> vec{};
    using in_t = std::array<minimize::floating_t, 2>;
    using m_t = minimize::Measurement<2>;
    vec.emplace_back(m_t{in_t{-10, -10}, 285.0});
    vec.emplace_back(m_t{in_t{-9, -9}, 231.8});
    vec.emplace_back(m_t{in_t{-8, -8}, 184.2});
    vec.emplace_back(m_t{in_t{-7, -7}, 142.2});
    vec.emplace_back(m_t{in_t{-6, -6}, 105.80000000000001});
    vec.emplace_back(m_t{in_t{-5, -5}, 75.0});
    vec.emplace_back(m_t{in_t{-4, -4}, 49.8});
    vec.emplace_back(m_t{in_t{-3, -3}, 30.200000000000003});
    vec.emplace_back(m_t{in_t{-2, -2}, 16.2});
    vec.emplace_back(m_t{in_t{-1, -1}, 7.8});
    vec.emplace_back(m_t{in_t{0, 0}, 5.0});
    vec.emplace_back(m_t{in_t{1, 1}, 7.8});
    vec.emplace_back(m_t{in_t{2, 2}, 16.2});
    vec.emplace_back(m_t{in_t{3, 3}, 30.200000000000003});
    vec.emplace_back(m_t{in_t{4, 4}, 49.8});
    vec.emplace_back(m_t{in_t{5, 5}, 75.0});
    vec.emplace_back(m_t{in_t{6, 6}, 105.80000000000001});
    vec.emplace_back(m_t{in_t{7, 7}, 142.2});
    vec.emplace_back(m_t{in_t{8, 8}, 184.2});
    vec.emplace_back(m_t{in_t{9, 9}, 231.8});
    vec.emplace_back(m_t{in_t{10, 10}, 285.0});
    return vec;
}

inline minimize::MeasurementVector<2> create_noisy_test_data_saddle() {
    // measurements for 0.5, 1.0, 1.3, 5.0
    minimize::MeasurementVector<2> vec{};
    using in_t = std::array<minimize::floating_t, 2>;
    using m_t = minimize::Measurement<2>;
    vec.emplace_back(m_t{in_t{-10, -10}, 285.1});
    vec.emplace_back(m_t{in_t{-9, -9}, 231.7});
    vec.emplace_back(m_t{in_t{-8, -8}, 184.2});
    vec.emplace_back(m_t{in_t{-7, -7}, 142.4});
    vec.emplace_back(m_t{in_t{-6, -6}, 105.80000000000001});
    vec.emplace_back(m_t{in_t{-5, -5}, 74.8});
    vec.emplace_back(m_t{in_t{-4, -4}, 49.8});
    vec.emplace_back(m_t{in_t{-3, -3}, 30.200000000000003});
    vec.emplace_back(m_t{in_t{-2, -2}, 16.2});
    vec.emplace_back(m_t{in_t{-1, -1}, 7.8});
    vec.emplace_back(m_t{in_t{0, 0}, 5.0});
    vec.emplace_back(m_t{in_t{1, 1}, 7.8});
    vec.emplace_back(m_t{in_t{2, 2}, 16.2});
    vec.emplace_back(m_t{in_t{3, 3}, 30.200000000000003});
    vec.emplace_back(m_t{in_t{4, 4}, 49.8});
    vec.emplace_back(m_t{in_t{5, 5}, 75.2});
    vec.emplace_back(m_t{in_t{6, 6}, 105.80000000000001});
    vec.emplace_back(m_t{in_t{7, 7}, 142.2});
    vec.emplace_back(m_t{in_t{8, 8}, 184.2});
    vec.emplace_back(m_t{in_t{9, 9}, 231.8});
    vec.emplace_back(m_t{in_t{10, 10}, 284.9});
    return vec;
}

#endif /* MINIMIZE_COMMON_INCLUDED_HPP */
