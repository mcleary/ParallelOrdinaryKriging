
#include "KrigingSerial.h"
#include "KrigingCommon.h"
#include "Timer.h"

#include <iostream>
#include <cmath>
#include <numeric>

using namespace std;

void Serialkriging::SerialKrigFit(const PointVector &InputPoints, int NumberOfPoints, int LagsCount)
{
    this->NumberOfPoints = NumberOfPoints;
    
	auto MinMaxXPair = minmax_element(InputPoints.begin(), InputPoints.end(), [](const PointXYZ& Point1, const PointXYZ& Point2)
	{
		return Point1.x < Point2.x;
	});
	auto MinMaxYPair = minmax_element(InputPoints.begin(), InputPoints.end(), [](const PointXYZ& Point1, const PointXYZ& Point2)
	{
		return Point1.y < Point2.y;
	});
	auto MinMaxZPair = minmax_element(InputPoints.begin(), InputPoints.end(), [](const PointXYZ& Point1, const PointXYZ& Point2)
	{
		return Point1.z < Point2.z;
	});
    
    MinPoint.x = (*MinMaxXPair.first).x;
    MinPoint.y = (*MinMaxYPair.first).y;
    MinPoint.z = (*MinMaxZPair.first).z;
    
    MaxPoint.x = (*MinMaxXPair.second).x;
    MaxPoint.y = (*MinMaxYPair.second).y;
    MaxPoint.z = (*MinMaxZPair.second).z;
    
    cout << "MinPoint: " << MinPoint << endl;
    cout << "MaxPoint: " << MaxPoint << endl;
    
    const float Cutoff = Dist(MaxPoint.x, MaxPoint.y, MinPoint.x, MinPoint.y) / 3.0f;
    auto LagRanges = GetLagRanges(Cutoff, LagsCount);
    
    Eigen::MatrixXf DistancesMatrix(NumberOfPoints, NumberOfPoints);

    for(int i = 0; i < NumberOfPoints; i++)
    {
        const auto& PointI = InputPoints[i];
        for(int j = 0; j < NumberOfPoints; ++j)
        {
            const auto& PointJ = InputPoints[j];
            DistancesMatrix(i, j) = Dist(PointI.x, PointI.y, PointJ.x, PointJ.y);
        }
    }
    
    cout << "Computing Semivariogram ... " << flush;
    
    vector<float> EmpiricalSemivariogramX;
    vector<float> EmpiricalSemivariogramY;
    
    for (int LagIndex = 0; LagIndex < LagsCount; ++LagIndex)
    {
        const float RangeMin = LagRanges[LagIndex * 2 + 0];
        const float RangeMax = LagRanges[LagIndex * 2 + 1];
        
        vector<float> DistValues;
        vector<float> SemivarValues;
        
        for(int i = 0; i < NumberOfPoints; i++)
        {
            for(int j = 0; j < NumberOfPoints; ++j)
            {
                auto DistIJ = DistancesMatrix(i, j);
                
                if(RangeMin < DistIJ && DistIJ < RangeMax)
                {
                    const auto& PointIValue = InputPoints[i].z;
                    const auto& PointJValue = InputPoints[j].z;
                    
                    auto SemivarValue = pow(PointIValue - PointJValue, 2);
                    
					DistValues.push_back(DistIJ);
					SemivarValues.push_back(SemivarValue);
				}
            }
        }
        
        float AvgDistance = accumulate(DistValues.begin(), DistValues.end(), 0.0f);
        float AvgSemivar = accumulate(SemivarValues.begin(), SemivarValues.end(), 0.0f);
        
        AvgDistance /= DistValues.size();
        AvgSemivar /= SemivarValues.size();
        
		{
			EmpiricalSemivariogramX.push_back(AvgDistance);
			EmpiricalSemivariogramY.push_back(0.5f * AvgSemivar);
		}		
    }
    
    cout << "done" << endl;
    
    cout << "Fitting Linear Model to Semivariogram ... " << flush;
    auto LinearModel = LinearModelFit(EmpiricalSemivariogramX, EmpiricalSemivariogramY);
    
    const float LinearModelA = LinearModel.first;
    const float LinearModelB = LinearModel.second;
    
    Nugget = LinearModelA;
    Range = *max_element(EmpiricalSemivariogramX.begin(), EmpiricalSemivariogramX.end());
    Sill = Nugget + LinearModelB * Range;
    cout << "done" << endl;
    
    cout << "Nugget: " << Nugget << endl;
    cout << "Range : " << Range << endl;
    cout << "Sill  : " << Sill << endl;
    
    cout << "Calculating Covariance Matrix ..." << flush;
    
    Eigen::MatrixXf CovarianceMatrix(NumberOfPoints + 1, NumberOfPoints + 1);
    CovarianceMatrix.fill(1.0f);
    CovarianceMatrix(NumberOfPoints, NumberOfPoints) = 0.0f;
    
    for(int i = 0; i < NumberOfPoints; i++)
    {
        for(int j = 0; j < NumberOfPoints; ++j)
        {
            auto DistIJ = DistancesMatrix(i, j);
            CovarianceMatrix(i, j) = SphericalModel(DistIJ, Nugget, Range, Sill);
        }
    }
    
    cout << "done" << endl;
    
    cout << "Inverting Covariance Matrix ..." << flush;
    InvCovMatrix = CovarianceMatrix.cast<double>();
    InvCovMatrix = InvCovMatrix.inverse();
    cout << "done" << endl;
}

PointVector Serialkriging::SerialKrigPred(const PointVector &InputPoints, int GridSize)
{
    cout << "Predicting ... " << flush;
    
    PointVector Grid(GridSize * GridSize);
    float GridDeltaX = (MaxPoint.x - MinPoint.x) / GridSize;
    float GridDeltaY = (MaxPoint.y - MinPoint.y) / GridSize;
    
    Eigen::VectorXd ZValues(NumberOfPoints + 1);
    Eigen::VectorXd RValues(NumberOfPoints + 1);
    for(int i = 0; i < NumberOfPoints; ++i)
    {
        ZValues[i] = InputPoints[i].z;
    }
    ZValues[NumberOfPoints] = 1.0;
    RValues[NumberOfPoints] = 1.0;

    Eigen::VectorXd InvAXR(NumberOfPoints + 1);
    
    for (int i = 0; i < GridSize; ++i)
    {
        cout << i << " " << flush;

        for (int j = 0; j < GridSize; ++j)
        {
            float GridX = MinPoint.x + i * GridDeltaX;
            float GridY = MinPoint.y + j * GridDeltaY;
            
            for(int PIndex = 0; PIndex < NumberOfPoints; PIndex++)
            {
                const auto& Point = InputPoints[PIndex];
                auto UDist = Dist(GridX, GridY, Point.x, Point.y);
                RValues[PIndex] = SphericalModel(UDist, Nugget, Range, Sill);
            }
            
            InvAXR = InvCovMatrix * RValues;
            double GridZ = InvAXR.dot(ZValues);
            
            Grid[i + j * GridSize] = PointXYZ(GridX, GridY, GridZ);			
        }
    }
    
    cout << "done" << endl;
    
    return Grid;
}
