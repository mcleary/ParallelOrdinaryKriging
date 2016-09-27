#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void MatVecMulCPUKernel(global double* A, global double* x, int N, global double* y)
{
    int Index = get_global_id(0);

    global double* a = &A[Index * N];

	double Sum = 0.0;
	for(int j = 0; j < N; j++)
    {
		Sum += a[j] * x[j];
	}

	y[Index] = Sum;
}

kernel void VecMul(global double* a, global double* b, global double* Result, int N)
{
    int Index = get_global_id(0);   
    
    Result[Index] = a[Index] * b[Index];
}

#define ROW_DIM 0
#define COL_DIM 1

// P threads per row compute 1/P-th of each dot product.
// WORK has P columns and get_local_size(0) rows.
// Kernel from: http://www.bealto.com/gpu-gemv_v2.html
kernel void MatVecMulGPUKernel(
	global const double * a,
	global const double * x,
	global double * y,
	local double * work,
	int m, int n)
{
	// Compute partial dot product
	double sum = 0.0;
	for (int k = get_global_id(COL_DIM); k < n; k += get_global_size(COL_DIM))
	{
		sum += a[get_global_id(ROW_DIM) + m*k] * x[k];
	}

	// Each thread stores its partial sum in WORK
	int rows = get_local_size(ROW_DIM); // rows in group
	int cols = get_local_size(COL_DIM); // initial cols in group
	int ii = get_local_id(ROW_DIM); // local row index in group, 0<=ii<rows
	int jj = get_local_id(COL_DIM); // block index in column, 0<=jj<cols

	work[ii + rows * jj] = sum;

	barrier(CLK_LOCAL_MEM_FENCE); // sync group

	// Reduce sums in log2(cols) steps
	while (cols > 1)
	{
		cols >>= 1;
		if (jj < cols)
		{
			work[ii + rows*jj] += work[ii + rows*(jj + cols)];
		}		
		barrier(CLK_LOCAL_MEM_FENCE); // sync group
	}

	// Write final result in Y
	if (jj == 0)
	{
		y[get_global_id(ROW_DIM)] = work[ii];
	}
}
