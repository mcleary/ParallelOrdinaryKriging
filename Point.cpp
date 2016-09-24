//
//  Point.cpp
//  ParallelDTM
//
//  Created by Thales Sabino on 8/25/16.
//  Copyright © 2016 Thales Sabino. All rights reserved.
//

#include "Point.h"

using namespace std;

PointXYZ::PointXYZ(float _x, float _y, float _z) : x(_x), y(_y), z(_z)
{}

std::ostream& operator<< (std::ostream& out, const PointXYZ& p)
{
    out << p.x << " " << p.y << " " << p.z;
    return out;
}