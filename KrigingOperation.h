
#include "ComputePlatform.h"

#include "Eigen/Dense"

class KrigingOperation
{
public:
    explicit KrigingOperation(ComputePlatform& Platform);

	void KrigFit(const PointVector& InputPoints, int NumberOfPoints, int LagsCount = 10);
	PointVector KrigPred(const PointVector& InputPoints, int GridSize);

	PointXYZ MinPoint;
	PointXYZ MaxPoint;
	int NumberOfPoints;

	float Nugget;
	float Range;
	float Sill;
	Eigen::MatrixXd InvCovMatrix;

private:

    cl::Program KrigingProgram;
    ComputePlatform& ThePlatform;
};
