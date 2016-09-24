

#include "FillBufferOperation.h"

#include <iostream>

using namespace std;

FillBufferOperation::FillBufferOperation(ComputePlatform& Platform) :
    ThePlatform(Platform)
{
    FillBufferProgram = ThePlatform.CreateProgram("kernels/Buffers.cl");
}

cl::Event FillBufferOperation::FillDoubleBuffer(cl::CommandQueue Queue, cl::Buffer Buffer, double Value, int Count)
{
    DEBUG_OPERATION;
    
    auto FillBufferKernel = cl::make_kernel<cl::Buffer, double>(FillBufferProgram, "FillDoubleBuffer");
    
    return FillBufferKernel(cl::EnqueueArgs(Queue, cl::NDRange(Count)), Buffer, Value);
}

cl::Event FillBufferOperation::FillFloatBuffer(cl::CommandQueue Queue, cl::Buffer Buffer, float Value, int Count)
{
    DEBUG_OPERATION;
    
    auto FillBufferKernel = cl::make_kernel<cl::Buffer, float>(FillBufferProgram, "FillFloatBuffer");
    
    return FillBufferKernel(cl::EnqueueArgs(Queue, cl::NDRange(Count)), Buffer, Value);
}

cl::Event FillBufferOperation::FillIntBuffer(cl::CommandQueue Queue, cl::Buffer Buffer, int Value, int Count)
{
    DEBUG_OPERATION;
    
    auto FillBufferKernel = cl::make_kernel<cl::Buffer, int>(FillBufferProgram, "FillIntBuffer");
    
    return FillBufferKernel(cl::EnqueueArgs(Queue, cl::NDRange(Count)), Buffer, Value);
}

cl::Event FillBufferOperation::FillDoubleBuffer(cl::Buffer Buffer, double Value, int Count)
{
    auto Queue = ThePlatform.GetNextCommandQueue();
    
    return FillDoubleBuffer(Queue, Buffer, Value, Count);
}

cl::Event FillBufferOperation::FillFloatBuffer(cl::Buffer Buffer, float Value, int Count)
{
    auto Queue = ThePlatform.GetNextCommandQueue();
    
    return FillFloatBuffer(Queue, Buffer, Value, Count);
}

cl::Event FillBufferOperation::FillIntBuffer(cl::Buffer Buffer, int Value, int Count)
{
    auto Queue = ThePlatform.GetNextCommandQueue();
    
    return FillIntBuffer(Queue, Buffer, Value, Count);
}