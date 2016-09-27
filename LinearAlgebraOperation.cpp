
#include "LinearAlgebraOperation.h"

using namespace std;

LinearAlgebraOperation::LinearAlgebraOperation(ComputePlatform& Platform) :
    ThePlatform(Platform),
    ReduceOperation(Platform)
{
    LinearAlgebraProgram = ThePlatform.CreateProgram("kernels/LinearAlgebra.cl");    
}

cl::Event LinearAlgebraOperation::MatVecMul(cl::CommandQueue Queue, cl::Buffer MatrixBuffer, cl::Buffer VectorBuffer, cl::Buffer ResultBuffer, int Count)
{
	auto Device = Queue.getInfo<CL_QUEUE_DEVICE>();
	auto DeviceType = Device.getInfo<CL_DEVICE_TYPE>();

	DEBUG_OPERATION;

	switch (DeviceType)
	{
	case CL_DEVICE_TYPE_CPU:
		return MatVecMulCPU(Queue, MatrixBuffer, VectorBuffer, ResultBuffer, Count);
	case CL_DEVICE_TYPE_GPU:
		return MatVecMulGPU(Queue, MatrixBuffer, VectorBuffer, ResultBuffer, Count);
	default:
		break;
	}

	return cl::Event();
}

double LinearAlgebraOperation::DotProduct(cl::CommandQueue Queue, cl::Buffer A, cl::Buffer B, int Count, cl::Buffer CacheBuffer)
{
    DEBUG_OPERATION;            
	
	if (CacheBuffer.getInfo<CL_MEM_SIZE>() != Count * sizeof(double))
	{
		throw runtime_error("CacheBuffer must have at least " + to_string(Count * sizeof(double)) + " bytes");
	}

	auto VecMulKernel = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>(LinearAlgebraProgram, "VecMul");

    auto VecMulEvent = VecMulKernel(cl::EnqueueArgs(Queue, cl::NDRange(Count)), A, B, CacheBuffer, Count);
    
    return ReduceOperation.ReduceDouble(Queue, CacheBuffer, Count, ReductionOp::Sum);
}

cl::Event LinearAlgebraOperation::MatVecMulCPU(cl::CommandQueue Queue, cl::Buffer MatrixBuffer, cl::Buffer VectorBuffer, cl::Buffer ResultBuffer, int Count)
{
	auto MatVecMulKernel = cl::make_kernel<
		cl::Buffer, 
		cl::Buffer, 
		int, 
		cl::Buffer
	>(LinearAlgebraProgram, "MatVecMulCPUKernel");

	return MatVecMulKernel(
		cl::EnqueueArgs(Queue, cl::NDRange(Count)),
		MatrixBuffer,
		VectorBuffer,
		Count,
		ResultBuffer
	);
}

cl::Event LinearAlgebraOperation::MatVecMulGPU(cl::CommandQueue Queue, cl::Buffer MatrixBuffer, cl::Buffer VectorBuffer, cl::Buffer ResultBuffer, int Count)
{
	auto MatVecMulKernel = cl::make_kernel<
		cl::Buffer,
		cl::Buffer,
		cl::Buffer,
		cl::LocalSpaceArg,
		int,
		int
	>(LinearAlgebraProgram, "MatVecMulGPUKernel");

	const int PThreads = 8;

	int WorkGroupCount = Count;
	while (WorkGroupCount % PThreads != 0)
	{
		WorkGroupCount++;
	}

	int WorkItemCount = 64;
	while (WorkGroupCount % WorkItemCount != 0)
	{
		WorkItemCount >>= 1;
	}

	return MatVecMulKernel(
		cl::EnqueueArgs(
			Queue, 
			cl::NDRange(WorkGroupCount, PThreads), 
			cl::NDRange(WorkItemCount, PThreads)
		),
		MatrixBuffer,
		VectorBuffer,
		ResultBuffer,
		cl::Local(WorkItemCount * PThreads * sizeof(double)),
		Count,
		Count
	);
}