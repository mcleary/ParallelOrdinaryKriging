//
//  ComputePlatform.hpp
//  ParallelDTM
//
//  Created by Thales Sabino on 8/25/16.
//  Copyright © 2016 Thales Sabino. All rights reserved.
//

#pragma once

#define __CL_ENABLE_EXCEPTIONS
#ifdef _WIN32
#	include <CL/cl.hpp>
#else
#	include "CL/cl.hpp"
#endif

#include <vector>
#include <mutex>
#include <map>

#include "Point.h"

#define PRINT_DEVICE_NAMES 0

#define DEBUG_OPERATION \
    ThePlatform.PrintDeviceName(__FUNCTION__, Queue)

class ComputePlatform
{
public:
    static int GetPlatformCount();
    
    explicit ComputePlatform(int PlatformIndex = 0, int NumDevices = -1);
    
    cl::Program CreateProgram(const std::string& SourceFilepath);
    
    cl::CommandQueue GetNextCommandQueue();	
    
    void PrintDeviceName(const std::string& Description, cl::CommandQueue Queue);

	void RecordEvent(const std::vector<std::string>& Tags, cl::Event Event);
	void RecordTime(const std::vector<std::string>& Tags, long int Time);
    
public:
    cl::Platform                    Platform;
    cl::Context                     Context;

	std::vector<cl::Device>			Devices;
	std::vector<cl::CommandQueue>   CommandQueues;
	std::mutex						CommandQueuesMutex;

	std::mutex						RecordEventMutex;
	bool							bProfile;
	std::map<std::string, long int> ProfilingMap;
};

std::ostream& operator<< (std::ostream& out, const ComputePlatform& aComputingPlatorm);




