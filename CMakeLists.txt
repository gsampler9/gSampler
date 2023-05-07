cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(GS LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_torch.py FIND_TORCH_PY)
set(PYTHON_INTERP python)
message(STATUS "Using Python interpreter: ${PYTHON_INTERP}")
execute_process(
  COMMAND ${PYTHON_INTERP} ${FIND_TORCH_PY}
  OUTPUT_VARIABLE TORCH_PREFIX_VER
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "find_torch.py output: ${TORCH_PREFIX_VER}")
list(GET TORCH_PREFIX_VER 0 TORCH_PREFIX)
list(GET TORCH_PREFIX_VER 1 TORCH_VER)
message(STATUS "Configuring for PyTorch ${TORCH_VER}")
set(Torch_DIR "${TORCH_PREFIX}/Torch")
find_package(Torch REQUIRED)
list(APPEND GS_EXTERNAL_LIB ${TORCH_LIBRARIES})

# CUDA and Thrust
add_subdirectory(third_party/thrust)
set(THRUST_INCLUDE_CUB_CMAKE ON)
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)
cuda_include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third_party/thrust")
cuda_include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third_party/thrust/dependencies/cub")
message(STATUS "Thrust ${THRUST_VERSION}")

# GLOG
SET(WITH_GFLAGS OFF)
SET(WITH_GTEST OFF)
add_subdirectory(third_party/glog)
list(APPEND GS_EXTERNAL_LIB glog::glog)

file(GLOB GS_SRC
  src/*.cc
  src/cuda/*.cu
  src/cuda/fusion/*.cu
)

add_library(gs SHARED ${GS_SRC})
target_link_libraries(gs ${GS_EXTERNAL_LIB})
target_link_libraries(gs Thrust)
set_target_properties(gs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gs PROPERTIES CUDA_STANDARD 14)
set_target_properties(gs PROPERTIES POSITION_INDEPENDENT_CODE ON)

enable_testing()
add_subdirectory(third_party/googletest)
include_directories(third_party/googletest/googletest/include)
include_directories("src")
file(GLOB_RECURSE TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/tests/*.cc)
add_executable(runUnitTests ${TEST_SRC_FILES})
target_link_libraries(runUnitTests gs gtest gtest_main)
add_test(UnitTests runUnitTests)

