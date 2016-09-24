

#define OP_MIN 0
#define OP_MAX 1
#define OP_SUM 2

inline float NeutralElement(const int Operation)
{
	switch (Operation)
	{
	case OP_MIN:
		return INFINITY;
	case OP_MAX:
		return -INFINITY;
	case OP_SUM:
		return 0.0f;
    default:
        break;
	}
    return 0.0f;
}

inline float Reduce(float Accumulator, float Element, const int Operation)
{
	switch (Operation)
	{
	case OP_MIN:
		Accumulator = (Accumulator < Element) ? Accumulator : Element;
		break;
	case OP_MAX:
		Accumulator = (Accumulator > Element) ? Accumulator : Element;
		break;
	case OP_SUM:
		Accumulator += Element;
	default:
		break;
	}

	return Accumulator;
}

kernel void ReduceKernel(global   float*  buffer,
                         const    int     block,
                         const    int     length,
                         global   float*  result,
                         const    int	  op)
{
	int global_index = get_global_id(0) * block;
	int upper_bound = (get_global_id(0) + 1) * block;

	if (upper_bound > length)
	{
		upper_bound = length;
	}

	float accumulator = NeutralElement(op);

	while (global_index < upper_bound)
	{
		float element = buffer[global_index];
		accumulator = Reduce(accumulator, element, op);
		global_index++;
	}

	result[get_group_id(0)] = accumulator;
}

kernel void TwoStageReduceKernel(global float*  buffer,
                                 local  float*  scratch,
                                 const  int     length,
                                 global float*  result,
                                 const int		op)
{
    int global_index = get_global_id(0);
    float accumulator = NeutralElement(op);	
    
    // Loop sequentially over chunks of input vector
    while (global_index < length)
    {
        float element = buffer[global_index];
		accumulator = Reduce(accumulator, element, op);
        global_index += get_global_size(0);
    }	
    
    // Perform parallel reduction
    int local_index = get_local_id(0);
    scratch[local_index] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2)
    {
        if (local_index < offset)
        {
            float other = scratch[local_index + offset];
            float mine = scratch[local_index];
			scratch[local_index] = Reduce(mine, other, op);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_index == 0)
    {
        result[get_group_id(0)] = scratch[0];
    }
}



struct PointXYZ
{
    float x;
    float y;
    float z;
};

inline struct PointXYZ ReducePoint(struct PointXYZ Accumulator, struct PointXYZ Element, const int Operation)
{
	Accumulator.x = Reduce(Accumulator.x, Element.x, Operation);
	Accumulator.y = Reduce(Accumulator.y, Element.y, Operation);
	Accumulator.z = Reduce(Accumulator.z, Element.z, Operation);
	return Accumulator;
}

kernel void ReducePointsKernel(global struct PointXYZ* buffer,
                               const int block,
                               const int length,
                               global struct PointXYZ* result,
                               const int op)
{
	int global_index = get_global_id(0) * block;
	int upper_bound = (get_global_id(0) + 1) * block;

	if (upper_bound > length)
	{
		upper_bound = length;
	}

	float Neutral = NeutralElement(op);
	struct PointXYZ accumulator = { Neutral, Neutral, Neutral };

	while (global_index < upper_bound)
	{
		struct PointXYZ element = buffer[global_index];

		accumulator = ReducePoint(accumulator, element, op);

		global_index++;
	}

	result[get_group_id(0)] = accumulator;
}


kernel void TwoStageReducePointKernel(global struct PointXYZ*  buffer,
                                      local  struct PointXYZ*  scratch,
                                      const  int               length,
                                      global struct PointXYZ*  result,
                                      const  int			   op)
{
	int global_index = get_global_id(0);

	float Neutral = NeutralElement(op);
	struct PointXYZ accumulator = { Neutral, Neutral, Neutral };

	// Loop sequentially over chunks of input vector
	while (global_index < length)
	{
		struct PointXYZ element = buffer[global_index];

		accumulator = ReducePoint(accumulator, element, op);

		global_index += get_global_size(0);
	}

	// Perform parallel reduction
	int local_index = get_local_id(0);

	scratch[local_index] = accumulator;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int offset = get_local_size(0) / 2; offset > 0; offset >>= 1)
	{
		if (local_index < offset)
		{
			struct PointXYZ other = scratch[local_index + offset];
			struct PointXYZ mine = scratch[local_index];

			scratch[local_index] = ReducePoint(mine, other, op);
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (local_index == 0)
	{
		result[get_group_id(0)] = scratch[0];
	}
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void ReduceDoubleKernel(global   double*  buffer,
                               const    int     block,
                               const    int     length,
                               global   double*  result,
                               const    int	  op)
{
    int global_index = get_global_id(0) * block;
    int upper_bound = (get_global_id(0) + 1) * block;
    
    if (upper_bound > length)
    {
        upper_bound = length;
    }
    
    double accumulator = NeutralElement(op);
    
    while (global_index < upper_bound)
    {
        float element = buffer[global_index];
        accumulator = Reduce(accumulator, element, op);
        global_index++;
    }
    
    result[get_group_id(0)] = accumulator;
}

kernel void TwoStageReduceDoubleKernel(global double*  buffer,
                                       local  double*  scratch,
                                       const  int     length,
                                       global double*  result,
                                       const int		op)
{
    int global_index = get_global_id(0);
    double accumulator = NeutralElement(op);
    
    // Loop sequentially over chunks of input vector
    while (global_index < length)
    {
        double element = buffer[global_index];
        accumulator = Reduce(accumulator, element, op);
        global_index += get_global_size(0);
    }
    
    // Perform parallel reduction
    int local_index = get_local_id(0);
    scratch[local_index] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2)
    {
        if (local_index < offset)
        {
            double other = scratch[local_index + offset];
            double mine = scratch[local_index];
            scratch[local_index] = Reduce(mine, other, op);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_index == 0)
    {
        result[get_group_id(0)] = scratch[0];
    }
}
