
#include "ComputePlatform.h"

class DistancesMatrixOperation
{
public:
	explicit DistancesMatrixOperation(ComputePlatform& ThePlatform);

    cl::Event ComputeMatrix(cl::Buffer InputBuffer, int NumberOfPoints, cl::Buffer OutputBuffer);

public:
	ComputePlatform& ThePlatform;
	cl::Program		 DistancesMatrixProgram;
};