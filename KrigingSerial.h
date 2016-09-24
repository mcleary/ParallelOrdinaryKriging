
#pragma once

#include "Point.h"

#include "Eigen/Dense"

class Serialkriging
{
public:
    void SerialKrigFit(const PointVector& InputPoints, int NumberOfPoints, int LagsCount);
    
    PointVector SerialKrigPred(const PointVector& InputPoints, int GridSize);
    
private:
    PointXYZ MinPoint;
    PointXYZ MaxPoint;
    int NumberOfPoints;
    
    float Nugget;
    float Range;
    float Sill;
    Eigen::MatrixXd InvCovMatrix;
};


