

kernel void FillFloatBuffer(global float* Buffer, float Value)
{
    int Index = get_global_id(0);
    Buffer[Index] = Value;
}

kernel void FillIntBuffer(global float* Buffer, int Value)
{
    int Index = get_global_id(0);
    Buffer[Index] = Value;
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void FillDoubleBuffer(global double* Buffer, double Value)
{
    int Index = get_global_id(0);
    Buffer[Index] = Value;
}