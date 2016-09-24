
#pragma once

#include "ComputePlatform.h"

enum class ReductionOp
{
	Min = 0,
	Max = 1,
	Sum = 2
};

class ReductionOperation
{
public:
	explicit ReductionOperation(ComputePlatform& ThePlatform);

	float Reduce(cl::Buffer InputBuffer, int Count, ReductionOp Operator);
	PointXYZ ReducePoints(cl::Buffer InputBuffer, int NumberOfPoints, ReductionOp Operator);
    
    float Reduce(cl::CommandQueue Queue, cl::Buffer InputBuffer, int Count, ReductionOp Operator);
    PointXYZ ReducePoints(cl::CommandQueue Queue, cl::Buffer InputBuffer, int NumberOfPoints, ReductionOp Operator);
    
    double ReduceDouble(cl::CommandQueue Queue, cl::Buffer InputBuffer, int Count, ReductionOp Operator);

private:
	float	 ReduceCPUKernel(cl::CommandQueue Queue, cl::Buffer InputBuffer, int Count, ReductionOp Operation);
	float	 ReduceGPUKernel(cl::CommandQueue Queue, cl::Buffer InputBuffer, int Count, ReductionOp Operation);
	PointXYZ ReducePointsCPUKernel(cl::CommandQueue Queue, cl::Buffer InputBuffer, int NumberOfPoints, ReductionOp Operation);
	PointXYZ ReducePointsGPUKernel(cl::CommandQueue Queue, cl::Buffer InputBuffer, int NumberOfPoints, ReductionOp Operation);
    
    double   ReduceCPUDoubleKernel(cl::CommandQueue Queue, cl::Buffer InputBuffer, int Count, ReductionOp Operation);
    double   ReduceGPUDoubleKernel(cl::CommandQueue Queue, cl::Buffer InputBuffer, int Count, ReductionOp Operation);

	ComputePlatform& ThePlatform;
	cl::Program ReductionProgram;

	std::map<cl::Device, cl::Buffer> ResultBuffers;
};