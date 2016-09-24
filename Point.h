//
//  Point.hpp
//  ParallelDTM
//
//  Created by Thales Sabino on 8/25/16.
//  Copyright © 2016 Thales Sabino. All rights reserved.
//

#pragma once

#include <ostream>
#include <vector>

struct PointXYZ
{
    float x;
    float y;
    float z;
    
    PointXYZ() = default;
    
    PointXYZ(float _x, float _y, float _z);
};

typedef std::vector<PointXYZ> PointVector;

std::ostream& operator<< (std::ostream& out, const PointXYZ& p);