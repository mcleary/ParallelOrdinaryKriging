
#pragma once

#include <vector>

std::vector<float> GetLagRanges(float Cutoff, int LagsCount);

std::pair<float, float> LinearModelFit(const std::vector<float>& X, const std::vector<float>& Y);

double SphericalModel(double h, double Nugget, double Range, double Sill);

template<typename T>
inline T Dist(T x0, T y0, T x1, T y1)
{
	return sqrt(pow(x0 - x1, 2) + pow(y0 - y1, 2));
}