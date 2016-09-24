
struct PointXYZ
{
	float x;
	float y;
	float z;
};

kernel void SemivariogramKernel(
                             global struct PointXYZ* Points,
                             global float* DistancesMatrix,
                             const int NumberOfPoints,
                             const float RangeMin,
                             const float RangeMax,
                             global float* Distances,
                             global float* SemivarValues,
                             global int* ValidValuesCount
                             )
{
    int i = get_global_id(0);
    
    struct PointXYZ CurrentPoint = Points[i];
    
    for (int j = 0; j < NumberOfPoints; j++)
    {
        float Dist = DistancesMatrix[i + j * NumberOfPoints];
        
        int bDistIsInRange = RangeMin < Dist && Dist < RangeMax;
        
        struct PointXYZ OtherPoint = Points[j];
        
        const float SemivarValue = pow(CurrentPoint.z - OtherPoint.z, 2);
        const int SemivarIndex = i + j * NumberOfPoints;
        
        Distances[SemivarIndex] = Dist * bDistIsInRange;
        SemivarValues[SemivarIndex] = SemivarValue * bDistIsInRange;
        
		if (bDistIsInRange)
		{
			atomic_inc(ValidValuesCount);
		}		
    }
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
inline double SphericalModel(double h, double Nugget, double Range, double Sill)
{
    if (h >= Range)
    {
        return Sill;
    }
    
    double H_Over_Range = h / Range;
    double H_Over_Range3 = H_Over_Range * H_Over_Range * H_Over_Range;
    
    return (Sill - Nugget) * (1.5 * H_Over_Range - 0.5 * H_Over_Range3) + Nugget;
}

kernel void CovarianceMatrixKernel(
	global float* DistancesMatrix,
	global float* CovMatrix,
	const int NumberOfPoints,
	const float Nugget,
	const float Range,
	const float Sill
)
{
	int i = get_global_id(0);

	for (int j = 0; j < NumberOfPoints; ++j)
	{
		float Dist = DistancesMatrix[i + j * NumberOfPoints];
        
		const int Index = i + j * (NumberOfPoints + 1);
		CovMatrix[Index] = SphericalModel(Dist, Nugget, Range, Sill);
	}
}

inline double Distance(double P1X, double P1Y, double P2X, double P2Y)
{
    double dx2 = (P1X - P2X) * (P1X - P2X);
    double dy2 = (P1Y - P2Y) * (P1Y - P2Y);
    return sqrt(dx2 + dy2);
}

kernel void PredictionCovariance(global struct PointXYZ* Points,
                                 global double* Result,
                                 double Px,
                                 double Py,
                                 double Nugget,
                                 double Range,
                                 double Sill)
{
    int Index = get_global_id(0);
    
    struct PointXYZ Point = Points[Index];
    
    double Dist = Distance(Point.x, Point.y, Px, Py);
    Result[Index] = SphericalModel(Dist, Nugget, Range, Sill);
}