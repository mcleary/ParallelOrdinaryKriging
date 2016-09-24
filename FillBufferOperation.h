
#pragma once

#include "ComputePlatform.h"


class FillBufferOperation
{
public:
    explicit FillBufferOperation(ComputePlatform& Platform);
    
    cl::Event FillDoubleBuffer(cl::CommandQueue Queue, cl::Buffer Buffer, double Value, int Count);
    cl::Event FillFloatBuffer(cl::CommandQueue Queue, cl::Buffer Buffer, float Value, int Count);
    cl::Event FillIntBuffer(cl::CommandQueue Queue, cl::Buffer Buffer, int Value, int Count);
    
    cl::Event FillDoubleBuffer(cl::Buffer Buffer, double Value, int Count);
    cl::Event FillFloatBuffer(cl::Buffer Buffer, float Value, int Count);
    cl::Event FillIntBuffer(cl::Buffer Buffer, int Value, int Count);
    
private:
    cl::Program FillBufferProgram;
    ComputePlatform& ThePlatform;
};