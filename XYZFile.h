//
//  XYZFile.h
//  ParallelDTM
//
//  Created by Thales Sabino on 8/25/16.
//  Copyright © 2016 Thales Sabino. All rights reserved.
//

#pragma once

#include <vector>

#include "Point.h"

PointVector ReadXYZFile(const std::string& Filepath);

void WriteXYZFile(const std::string& Filepath, PointVector Points);