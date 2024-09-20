cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(NEOHOOKEAN_CUDA LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(GPU_ARCHS "80")
if(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
  set(GPU_ARCHS "${GPU_ARCHS};80")
endif()

set(CPPIMPACT_VNTX_PATH "${CMAKE_SOURCE_DIR}/extern/NVTX/c")

find_package(LAPACK REQUIRED)

option(CPPIMPACT_DEBUG_MODE "debug" OFF)
if(CPPIMPACT_DEBUG_MODE MATCHES ON)
  add_definitions(-DCPPIMPACT_DEBUG_MODE)
endif()

# C++ target
add_executable(cpu_test test.cu)
target_link_libraries(cpu_test ${LAPACK_LIBRARIES})

# Add FEA_debug target
add_executable(FEA_debug FEA_debug.cu)
# target_link_libraries(FEA_debug ${LAPACK_LIBRARIES})

# GPU target
add_executable(gpu_test test.cu)
set_target_properties(gpu_test PROPERTIES CUDA_ARCHITECTURES ${GPU_ARCHS})
set_target_properties(gpu_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_test PRIVATE -DCPPIMPACT_CUDA_BACKEND)
target_compile_options(gpu_test PRIVATE -G)  # device debug
# target_link_libraries(gpu_test PRIVATE nvtx3-cpp)