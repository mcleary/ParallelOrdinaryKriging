
#include "KrigingCommon.h"

#include <numeric>
#include <cmath>

using namespace std;

vector<float> GetLagRanges(float Cutoff, int LagsCount)
{
    vector<float> LagRanges;
    
    for (int LagIndex = 1; LagIndex < LagsCount + 1; ++LagIndex)
    {
        float LagMin = (LagIndex - 1) * Cutoff / LagsCount;
        float LagMax = LagIndex * Cutoff / LagsCount;
        LagRanges.push_back(LagMin);
        LagRanges.push_back(LagMax);
    }
    
    return LagRanges;
}

pair<float, float> LinearModelFit(const vector<float>& X, const vector<float>& Y)
{
   	float MeanX = accumulate(X.begin(), X.end(), 0.0f) / X.size();
    float MeanY = accumulate(Y.begin(), Y.end(), 0.0f) / Y.size();
    
    float Sum1 = 0.0f;
    float Sum2 = 0.0f;
    for (int i = 0; i < X.size(); ++i)
    {
        Sum1 += (X[i] - MeanX) * (Y[i] - MeanY);
        Sum2 += pow(X[i] - MeanX, 2);
    }
    
    float b = Sum1 / Sum2;
    float a = MeanY - b * MeanX;
    
    return{ a, b };
}

double SphericalModel(double h, double Nugget, double Range, double Sill)
{
    if (h >= Range)
    {
        return Sill;
    }
    
    return (Sill - Nugget) * (1.5 * (h / Range) - 0.5 * pow(h / Range, 3)) + Nugget;
}
