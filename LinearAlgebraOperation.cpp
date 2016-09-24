
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
    DEBUG_OPERATION;

	auto MatVecMulKernel = cl::make_kernel<cl::Buffer, cl::Buffer, int, cl::Buffer>(LinearAlgebraProgram, "MatVecMul");
    
    return MatVecMulKernel(
		cl::EnqueueArgs(Queue, cl::NDRange(Count)), 
		MatrixBuffer, 
		VectorBuffer, 
		Count, 
		ResultBuffer		
	);

	//auto MatVecMulKernel = cl::make_kernel<
	//	cl::Buffer,
	//	cl::Buffer,
	//	cl::Buffer,
	//	cl::LocalSpaceArg,
	//	int,
	//	int
	//>(LinearAlgebraProgram, "gemv2");

	//int PThreads = 3;

	//return MatVecMulKernel(
	//	cl::EnqueueArgs(Queue, cl::NDRange(Count, PThreads), cl::NDRange(PThreads, 17)),
	//	MatrixBuffer,
	//	VectorBuffer,
	//	ResultBuffer,
	//	cl::Local(4096 * sizeof(double)),
	//	Count, 
	//	Count
	//	);
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