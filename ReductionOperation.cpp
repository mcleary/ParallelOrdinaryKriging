
#include "ReductionOperation.h"

#include <algorithm>
#include <limits>
#include <iostream>

template<class T>
T NeutralElement(ReductionOp Operation)
{
    switch (Operation)
    {
        case ReductionOp::Min:
            return std::numeric_limits<T>::max();
        case ReductionOp::Max:
            return -std::numeric_limits<T>::max();
        case ReductionOp::Sum:
            return static_cast<T>(0.0);
        default:
            break;
    }
    
    return static_cast<T>(0.0);
}

template<class T>
T Reduce(T Accumulator, T Element, ReductionOp Operation)
{
    switch (Operation)
    {
        case ReductionOp::Min:
            Accumulator = (Accumulator < Element) ? Accumulator : Element;
            break;
        case ReductionOp::Max:
            Accumulator = (Accumulator > Element) ? Accumulator : Element;
            break;
        case ReductionOp::Sum:
            Accumulator += Element;
            break;
        default:
            break;
    }
    
    return Accumulator;
}

template<class T>
T ReduceCPU(T* Elements, int Count, ReductionOp Operation)
{
    T Accumulator = NeutralElement<T>(Operation);
	for (int Index = 0; Index < Count; ++Index)
	{
		const T Element = Elements[Index];
        Accumulator = Reduce(Accumulator, Element, Operation);
	}

	return Accumulator;
}

static PointXYZ ReduceCPU(PointXYZ* Points, int Count, ReductionOp Operation)
{
    auto Neutral = NeutralElement<float>(Operation);
    
    PointXYZ Accumulator{ Neutral, Neutral, Neutral };
    
	for (int Index = 0; Index < Count; ++Index)
	{
		const PointXYZ& Point = Points[Index];
        
        Accumulator.x = Reduce(Accumulator.x, Point.x, Operation);
        Accumulator.y = Reduce(Accumulator.y, Point.y, Operation);
        Accumulator.z = Reduce(Accumulator.z, Point.z, Operation);
	}

	return Accumulator;
}

ReductionOperation::ReductionOperation(ComputePlatform& Platform) :
	ThePlatform(Platform)
{
	ReductionProgram = ThePlatform.CreateProgram("kernels/Reduction.cl");
}

float ReductionOperation::Reduce(cl::CommandQueue Queue, cl::Buffer InputBuffer, int Count, ReductionOp Operator)
{
    auto Device = Queue.getInfo<CL_QUEUE_DEVICE>();
    auto DeviceType = Device.getInfo<CL_DEVICE_TYPE>();
    
    DEBUG_OPERATION;
    
    switch (DeviceType)
    {
        case CL_DEVICE_TYPE_CPU:
            return ReduceCPUKernel(Queue, InputBuffer, Count, Operator);
        case CL_DEVICE_TYPE_GPU:
            return ReduceGPUKernel(Queue, InputBuffer, Count, Operator);
        default:
            break;
    }
    
    return 0.0f;
}

float ReductionOperation::Reduce(cl::Buffer InputBuffer, int Count, ReductionOp Operator)
{
	auto Queue = ThePlatform.GetNextCommandQueue();
    
    return Reduce(Queue, InputBuffer, Count, Operator);
}

PointXYZ ReductionOperation::ReducePoints(cl::CommandQueue Queue, cl::Buffer InputBuffer, int NumberOfPoints, ReductionOp Operator)
{
    auto Device = Queue.getInfo<CL_QUEUE_DEVICE>();
    auto DeviceType = Device.getInfo<CL_DEVICE_TYPE>();
    
    DEBUG_OPERATION;
    
    switch (DeviceType)
    {
        case CL_DEVICE_TYPE_CPU:
            return ReducePointsCPUKernel(Queue, InputBuffer, NumberOfPoints, Operator);
        case CL_DEVICE_TYPE_GPU:
            return ReducePointsGPUKernel(Queue, InputBuffer, NumberOfPoints, Operator);
        default:
            break;
    }
    
    return PointXYZ();
}

PointXYZ ReductionOperation::ReducePoints(cl::Buffer InputBuffer, int NumberOfPoints, ReductionOp Operator)
{
	auto Queue = ThePlatform.GetNextCommandQueue();
	
    return ReducePoints(Queue, InputBuffer, NumberOfPoints, Operator);
}

double ReductionOperation::ReduceDouble(cl::CommandQueue Queue, cl::Buffer InputBuffer, int Count, ReductionOp Operator)
{
    DEBUG_OPERATION;
    
    auto Device = Queue.getInfo<CL_QUEUE_DEVICE>();
    auto DeviceType = Device.getInfo<CL_DEVICE_TYPE>();
    
    DEBUG_OPERATION;
    
    switch (DeviceType)
    {
        case CL_DEVICE_TYPE_CPU:
            return ReduceCPUDoubleKernel(Queue, InputBuffer, Count, Operator);
        case CL_DEVICE_TYPE_GPU:
            return ReduceGPUDoubleKernel(Queue, InputBuffer, Count, Operator);
        default:
            break;
    }
    
    return 0.0;
}

float ReductionOperation::ReduceCPUKernel(cl::CommandQueue Queue, cl::Buffer InputBuffer, int Count, ReductionOp Operator)
{
	auto ReductionKernel = cl::make_kernel<cl::Buffer, int, int, cl::Buffer, int>(ReductionProgram, "ReduceKernel");

	auto Device = Queue.getInfo<CL_QUEUE_DEVICE>();
	auto NumberOfBlocks = Device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

	int BlockSize = Count / NumberOfBlocks;
	if (Count % NumberOfBlocks != 0)
	{
		BlockSize++;
	}

	const int ResultBufferSize = sizeof(float) * NumberOfBlocks;

	cl::Buffer ResultBuffer(ThePlatform.Context, CL_MEM_WRITE_ONLY, ResultBufferSize);

	cl::NDRange GlobalRange(NumberOfBlocks);
	cl::NDRange LocalRange(1);
	
	auto Event = ReductionKernel(cl::EnqueueArgs(Queue, GlobalRange, LocalRange), InputBuffer, BlockSize, Count, ResultBuffer, static_cast<int>(Operator));	

	float* PartialResult = (float*)Queue.enqueueMapBuffer(ResultBuffer, CL_TRUE, CL_MAP_READ, 0, ResultBufferSize);

	float Result = ReduceCPU(PartialResult, NumberOfBlocks, Operator);
    
    Queue.enqueueUnmapMemObject(ResultBuffer, PartialResult);

	ThePlatform.RecordEvent({ "TotalReduce", "ReduceCPU" }, Event);
    
    return Result;
}

float ReductionOperation::ReduceGPUKernel(cl::CommandQueue Queue, cl::Buffer InputBuffer, int Count, ReductionOp Operator)
{
	auto ReductionKernel = cl::make_kernel<cl::Buffer, cl::LocalSpaceArg, int, cl::Buffer, int>(ReductionProgram, "TwoStageReduceKernel");

	auto Device = Queue.getInfo<CL_QUEUE_DEVICE>();
	const int NumberOfComputeUnits = Device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

	const int NumberOfBlocks = NumberOfComputeUnits * 4;
	const int MaxBlockSize = 256;
    const int BlockSize = std::max(NumberOfBlocks % MaxBlockSize, 1);
    const int NumberOfGroups = std::max(NumberOfBlocks / BlockSize, 1);

	const int ResultBufferSize = sizeof(PointXYZ) * NumberOfGroups;

	cl::Buffer ResultBuffer(ThePlatform.Context, CL_MEM_WRITE_ONLY, ResultBufferSize);

	const int LocalMemSize = BlockSize * sizeof(PointXYZ);

	cl::NDRange GlobalRange(NumberOfBlocks);
	cl::NDRange LocalRange(BlockSize);
	auto Event = ReductionKernel(cl::EnqueueArgs(Queue, GlobalRange, LocalRange), InputBuffer, cl::Local(LocalMemSize), Count, ResultBuffer, static_cast<int>(Operator));	

	float* PartialResult = (float*)Queue.enqueueMapBuffer(ResultBuffer, CL_TRUE, CL_MAP_READ, 0, ResultBufferSize);

	float Result = ReduceCPU(PartialResult, NumberOfGroups, Operator);
    
    Queue.enqueueUnmapMemObject(ResultBuffer, PartialResult);

	ThePlatform.RecordEvent({ "Reduce", "ReduceGPU" }, Event);
    
    return Result;
}

PointXYZ ReductionOperation::ReducePointsCPUKernel(cl::CommandQueue Queue, cl::Buffer InputBuffer, int NumberOfPoints, ReductionOp Operator)
{
	auto ReductionKernel = cl::make_kernel<cl::Buffer, int, int, cl::Buffer, int>(ReductionProgram, "ReducePointsKernel");

	auto Device = Queue.getInfo<CL_QUEUE_DEVICE>();
	auto NumberOfBlocks = Device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

	int BlockSize = NumberOfPoints / NumberOfBlocks;
	if (NumberOfPoints % NumberOfBlocks != 0)
	{
		BlockSize++;
	}

	const int ResultBufferSize = sizeof(PointXYZ) * NumberOfBlocks;

	cl::Buffer ResultBuffer(ThePlatform.Context, CL_MEM_WRITE_ONLY, ResultBufferSize);

	cl::NDRange GlobalRange(NumberOfBlocks);
	cl::NDRange LocalRange(1);
	auto Event = ReductionKernel(cl::EnqueueArgs(Queue, GlobalRange, LocalRange), InputBuffer, BlockSize, NumberOfPoints, ResultBuffer, static_cast<int>(Operator));	

	PointXYZ* PartialResult = (PointXYZ*)Queue.enqueueMapBuffer(ResultBuffer, CL_TRUE, CL_MAP_READ, 0, ResultBufferSize);

	auto Result = ReduceCPU(PartialResult, NumberOfBlocks, Operator);
    
    Queue.enqueueUnmapMemObject(ResultBuffer, PartialResult);

	ThePlatform.RecordEvent({ "TotalReduce", "ReduceCPU" }, Event);
    
    return Result;
}

PointXYZ ReductionOperation::ReducePointsGPUKernel(cl::CommandQueue Queue, cl::Buffer InputBuffer, int NumberOfPoints, ReductionOp Operator)
{
	auto ReductionKernel = cl::make_kernel<cl::Buffer, cl::LocalSpaceArg, int, cl::Buffer, int>(ReductionProgram, "TwoStageReducePointKernel");

	auto Device = Queue.getInfo<CL_QUEUE_DEVICE>();
	const int NumberOfComputeUnits = Device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

	const int NumberOfBlocks = NumberOfComputeUnits * 4;
	const int MaxBlockSize = 256;
	const int BlockSize = std::max(NumberOfBlocks % MaxBlockSize, 1);
    const int NumberOfGroups = std::max(NumberOfBlocks / BlockSize, 1);

	// cout << "Number of Blocks: " << NumberOfBlocks << endl;
	// cout << "Block Size: " << BlockSize << endl;
	// cout << "Number of Groups: " << NumberOfGroups << endl;

	const int ResultBufferSize = sizeof(PointXYZ) * NumberOfGroups;

	cl::Buffer ResultBuffer(ThePlatform.Context, CL_MEM_WRITE_ONLY, ResultBufferSize);

	const int LocalMemSize = BlockSize * sizeof(PointXYZ);

	cl::NDRange GlobalRange(NumberOfBlocks);
	cl::NDRange LocalRange(BlockSize);
	auto Event = ReductionKernel(cl::EnqueueArgs(Queue, GlobalRange, LocalRange), InputBuffer, cl::Local(LocalMemSize), NumberOfPoints, ResultBuffer, static_cast<int>(Operator));

	PointXYZ* PartialResult = (PointXYZ*)Queue.enqueueMapBuffer(ResultBuffer, CL_TRUE, CL_MAP_READ, 0, ResultBufferSize);

	auto Result = ReduceCPU(PartialResult, NumberOfGroups, Operator);
    
    Queue.enqueueUnmapMemObject(ResultBuffer, PartialResult);

	ThePlatform.RecordEvent({ "TotalReduce", "ReduceGPU" }, Event);
    
    return Result;
}

double ReductionOperation::ReduceCPUDoubleKernel(cl::CommandQueue Queue, cl::Buffer InputBuffer, int Count, ReductionOp Operator)
{
    auto ReductionKernel = cl::make_kernel<cl::Buffer, int, int, cl::Buffer, int>(ReductionProgram, "ReduceDoubleKernel");
    
    auto Device = Queue.getInfo<CL_QUEUE_DEVICE>();
    auto NumberOfBlocks = Device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    
    int BlockSize = Count / NumberOfBlocks;
    if (Count % NumberOfBlocks != 0)
    {
        BlockSize++;
    }
    
    const int ResultBufferSize = sizeof(float) * NumberOfBlocks;
    
    cl::Buffer ResultBuffer(ThePlatform.Context, CL_MEM_WRITE_ONLY, ResultBufferSize);
    
    cl::NDRange GlobalRange(NumberOfBlocks);
    cl::NDRange LocalRange(1);
    auto Event = ReductionKernel(cl::EnqueueArgs(Queue, GlobalRange, LocalRange), InputBuffer, BlockSize, Count, ResultBuffer, static_cast<int>(Operator));
    
    double* PartialResult = (double*) Queue.enqueueMapBuffer(ResultBuffer, CL_TRUE, CL_MAP_READ, 0, ResultBufferSize);
    
    auto Result = ReduceCPU(PartialResult, NumberOfBlocks, Operator);
    
    Queue.enqueueUnmapMemObject(ResultBuffer, PartialResult);

	ThePlatform.RecordEvent({ "TotalReduce", "ReduceCPU" }, Event);
    
    return Result;
}

double ReductionOperation::ReduceGPUDoubleKernel(cl::CommandQueue Queue, cl::Buffer InputBuffer, int Count, ReductionOp Operator)
{
    auto ReductionKernel = cl::make_kernel<cl::Buffer, cl::LocalSpaceArg, int, cl::Buffer, int>(ReductionProgram, "TwoStageReduceDoubleKernel");
    
    auto Device = Queue.getInfo<CL_QUEUE_DEVICE>();
    const int NumberOfComputeUnits = Device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    
    const int NumberOfBlocks = NumberOfComputeUnits * 4;
    const int MaxBlockSize = 256;
    const int BlockSize = std::max(NumberOfBlocks % MaxBlockSize, 1);
    const int NumberOfGroups = std::max(NumberOfBlocks / BlockSize, 1);
    
    const int ResultBufferSize = sizeof(PointXYZ) * NumberOfGroups;
    
    cl::Buffer ResultBuffer(ThePlatform.Context, CL_MEM_WRITE_ONLY, ResultBufferSize);
    
    const int LocalMemSize = BlockSize * sizeof(PointXYZ);
    
    cl::NDRange GlobalRange(NumberOfBlocks);
    cl::NDRange LocalRange(BlockSize);
    auto Event = ReductionKernel(cl::EnqueueArgs(Queue, GlobalRange, LocalRange), InputBuffer, cl::Local(LocalMemSize), Count, ResultBuffer, static_cast<int>(Operator));
    
    double* PartialResult = (double*)Queue.enqueueMapBuffer(ResultBuffer, CL_TRUE, CL_MAP_READ, 0, ResultBufferSize);
    
    auto Result = ReduceCPU(PartialResult, NumberOfGroups, Operator);
    
    Queue.enqueueUnmapMemObject(ResultBuffer, PartialResult);

	ThePlatform.RecordEvent({ "TotalReduce", "ReduceGPU" }, Event);
    
    return Result;
}