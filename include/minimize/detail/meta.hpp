// Copyright (C) 2023 by domohuhn
// SPDX-License-Identifier: Zlib

#ifndef MINIMIZE_DETAIL_META_INCLUDED_HPP
#define MINIMIZE_DETAIL_META_INCLUDED_HPP

#include <array>
#include <cstdint>

namespace minimize {

using floating_t = double;

namespace detail {

template <std::size_t NumberOfParameters>
struct type_selection_helper {
    using type = std::array<floating_t, NumberOfParameters>;
};

template <>
struct type_selection_helper<1> {
    using type = floating_t;
};

}  // namespace detail
}  // namespace minimize

#endif /* MINIMIZE_DETAIL_META_INCLUDED_HPP */
