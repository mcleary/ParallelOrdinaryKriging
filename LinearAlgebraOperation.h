
#pragma once

#include "ComputePlatform.h"
#include "ReductionOperation.h"

class LinearAlgebraOperation
{
public:
    explicit LinearAlgebraOperation(ComputePlatform& Platform);
    
    cl::Event MatVecMul(cl::CommandQueue Queue, cl::Buffer MatrixBuffer, cl::Buffer VectorBuffer, cl::Buffer ResultBuffer, int Count);
    
    double DotProduct(cl::CommandQueue Queue, cl::Buffer A, cl::Buffer B, int Count, cl::Buffer CacheBuffer);
    
private:
    cl::Program      LinearAlgebraProgram;
	cl::Kernel       KernelTest;
	
    ComputePlatform&  ThePlatform;
    ReductionOperation ReduceOperation;
};