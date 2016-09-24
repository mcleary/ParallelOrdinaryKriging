
#include "DistancesMatrixOperation.h"

#include <iostream>

using namespace std;

DistancesMatrixOperation::DistancesMatrixOperation(ComputePlatform& Platform) :
	ThePlatform(Platform)
{
	DistancesMatrixProgram = ThePlatform.CreateProgram("kernels/DistancesMatrix.cl");
}

cl::Event DistancesMatrixOperation::ComputeMatrix(cl::Buffer InputBuffer, int NumberOfPoints, cl::Buffer OutputBuffer)
{
	auto DistancesMatrixKernel = cl::make_kernel<cl::Buffer, int, cl::Buffer>(DistancesMatrixProgram, "DistancesMatrixKernel");

	auto Queue = ThePlatform.GetNextCommandQueue();
    
    DEBUG_OPERATION;

	cl::NDRange GlobalRange(NumberOfPoints);
	return DistancesMatrixKernel(cl::EnqueueArgs(Queue, GlobalRange), InputBuffer, NumberOfPoints, OutputBuffer);
}