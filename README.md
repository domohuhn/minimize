# Minimize

[![CMake on multiple platforms](https://github.com/domohuhn/minimize/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/domohuhn/minimize/actions/workflows/cmake-multi-platform.yml)

A small C++ open source library that can be used to find minima of multi-dimensional functions.
The main selling point of this library is that it has no external dependencies and a permissive license.

## Quick start

The library is header only. All you need to do is to add the include folder to your include path.
Fitting a polynomial can be done via the following code snippet:

```c++
#include "minimize/minimize.hpp"

const auto data = read_measurement_data();
// create a polynomial of degree 1
minimize::Polynomial<1> poly{};
// set initial values
poly.set_parameter(0, 0.0);
poly.set_parameter(1, 0.0);
// fit
const auto chi2 = minimize::conjugate_gradient_descent(poly,data);

```

## Dependencies

You will need a C++ 11 compiler and the standard library.

Unit tests use the Catch2 framework. The framework is loaded on demand via cmake, but building the tests has to be enabled via an option when you call cmake.

## License
Everything is the repository uses the zlib license, see file "LICENSE".


