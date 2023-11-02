# Copyright (C) 2023 by domohuhn
# SPDX-License-Identifier: Zlib

cmake_minimum_required(VERSION 3.14)

project(minimize VERSION 1.0.0)

option(MINIMIZE_BUILD_TESTS "Build the unit tests for the minimize library" OFF)
option(MINIMIZE_BUILD_EXAMPLES "Build the examples for the minimize library" OFF)

add_library(minimize INTERFACE)

target_include_directories(minimize INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_compile_features(minimize INTERFACE cxx_std_11)

#
# Examples
#


if(MINIMIZE_BUILD_EXAMPLES)
  add_executable(minimize-linear-fit
    examples/linear_fit.cpp
  )
  target_link_libraries(minimize-linear-fit PRIVATE minimize)
endif()

#
# Tests
#

if(BUILD_TESTING)
  if(MINIMIZE_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)

    add_executable(test-minimize
      tests/wssr_test.cpp
      tests/function_gradient_test.cpp
      tests/find_minimum_on_line_test.cpp
      tests/steepest_descent_test.cpp
      tests/conjugate_gradient_test.cpp
      tests/polynomial_test.cpp
      tests/bootstrap_test.cpp
    )
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_compile_options(test-minimize PRIVATE -Wall -Wextra -pedantic -Werror)
    endif()
    target_link_libraries(test-minimize PRIVATE Catch2::Catch2WithMain minimize)
    catch_discover_tests(test-minimize)
  endif()
endif()

