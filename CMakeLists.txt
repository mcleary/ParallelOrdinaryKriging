cmake_minimum_required (VERSION 3.0)

project(ParallelDTM)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp")
SET(CMAKE_CXX_LINK_FLAGS "-fopenmp")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CLUSTER ON)

if(${CLUSTER})
  set(OpenCL_LIBRARY $ENV{AMDAPPSDKROOT}/lib/x86_64/libOpenCL.so)
endif()

find_package(OpenCL REQUIRED)

#include_directories(${OpenCL_INCLUDE_DIRS})
include_directories($ENV{AMDAPPSDKROOT}/include)

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH STRING FORCE)

add_executable(
  ParallelDTM
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

add_executable(
	Tests
	FillBufferOperation.cpp
	LinearAlgebraOperation.cpp
	ReductionOperation.cpp
	Point.cpp
	CommandLineParser.cpp
	ComputePlatform.cpp
	Timer.cpp
	Tests.cpp
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

INSTALL(TARGETS Tests DESTINATION ${CMAKE_INSTALL_PREFIX})
INSTALL(TARGETS ParallelDTM DESTINATION ${CMAKE_INSTALL_PREFIX})
INSTALL(FILES ${KERNELS_FILELIST} DESTINATION "kernels")
