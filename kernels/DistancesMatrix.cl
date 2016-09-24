
struct PointXYZ
{
	float x;
	float y;
	float z;
};

inline float Distance(struct PointXYZ Point1, struct PointXYZ Point2)
{
	return sqrt(pow(Point1.x - Point2.x, 2) +
				pow(Point1.y - Point2.y, 2));
}

kernel void DistancesMatrixKernel(global struct PointXYZ* InputData,
									 const int NumberOfPoints,								  						 
									 global float* Output)
{
	int GlobalIndex = get_global_id(0);
	
	struct PointXYZ CurrentPoint = InputData[GlobalIndex];

	for (int OtherIndex = 0; OtherIndex < NumberOfPoints; OtherIndex++)
	{
		struct PointXYZ OtherPoint = InputData[OtherIndex];
		Output[GlobalIndex + OtherIndex * NumberOfPoints] = Distance(CurrentPoint, OtherPoint);		
	}
}