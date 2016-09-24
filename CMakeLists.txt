cmake_minimum_required (VERSION 3.0)

project(ParallelDTM)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp")
SET(CMAKE_CXX_LINK_FLAGS "-fopenmp")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})

add_executable(
  ParallelOK
  CommandLineParser.cpp
  ComputePlatform.cpp
  DistancesMatrixOperation.cpp
  KrigingOperation.cpp
  ReductionOperation.cpp
  FillBufferOperation.cpp
  LinearAlgebraOperation.cpp
  XYZFile.cpp
  Point.cpp
  Timer.cpp
  main.cpp
)

target_link_libraries(ParallelDTM ${OpenCL_LIBRARIES})
target_link_libraries(Tests ${OpenCL_LIBRARIES})

set(
  KERNELS_FILELIST
  kernels/DistancesMatrix.cl
  kernels/Kriging.cl
  kernels/Reduction.cl
  kernels/Buffers.cl
  kernels/LinearAlgebra.cl
)


INSTALL(TARGETS ParallelOK DESTINATION ${CMAKE_INSTALL_PREFIX})
INSTALL(FILES ${KERNELS_FILELIST} DESTINATION "kernels")
