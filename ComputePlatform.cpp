//
//  ComputePlatform.cpp
//  ParallelDTM
//
//  Created by Thales Sabino on 8/25/16.
//  Copyright © 2016 Thales Sabino. All rights reserved.
//

#include "ComputePlatform.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>

#ifdef _WIN32
#	define STDCALL __stdcall
#else
#	define STDCALL
#endif

using namespace std;

void STDCALL OpenCLContextNotify(const char * info, const void *, ::size_t, void*)
{
    cout << info << endl;
}

static vector<cl::Platform> GetPlatforms()
{
    vector<cl::Platform> TempPlatforms;
    cl::Platform::get(&TempPlatforms);
    return TempPlatforms;
}

int ComputePlatform::GetPlatformCount()
{
    return static_cast<int>(GetPlatforms().size());
}

ComputePlatform::ComputePlatform(int PlatformIndex, int NumDevices)
{
    auto PlatformList = GetPlatforms();
    
    if(PlatformIndex >= static_cast<int>(PlatformList.size()))
    {
        throw std::runtime_error("Invalid platform index");
    }
    
    Platform = PlatformList.at(PlatformIndex);    
	Platform.getDevices(CL_DEVICE_TYPE_ALL, &Devices);

	if (Devices.empty())
	{
		throw std::runtime_error("No devices found for platform " + to_string(PlatformIndex));
	}

	if (NumDevices != -1)
	{
		while (Devices.size() > NumDevices)
		{
			Devices.pop_back();
		}
	}
    
    auto DeviceIsntValidPred = [](cl::Device aDevice)
    {
        return aDevice.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>() == 0;
    };
    Devices.erase(remove_if(Devices.begin(), Devices.end(), DeviceIsntValidPred), Devices.end());
    
    cl_context_properties ContextProperties[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(Platform)(),
        0
    };
    
    Context = cl::Context(Devices, ContextProperties, OpenCLContextNotify);	

	for (auto aDevice : Devices)
	{
		CommandQueues.emplace_back(Context, aDevice, CL_QUEUE_PROFILING_ENABLE);
	}
}

cl::Program ComputePlatform::CreateProgram(const std::string &SourceFilepath)
{
    ifstream KernelFile(SourceFilepath);
    
    if(!KernelFile.is_open())
    {
        throw std::runtime_error("Error opening " + SourceFilepath);
    }
    
    string KernelSource( istreambuf_iterator<char>(KernelFile), (istreambuf_iterator<char>()) );    
    cl::Program::Sources ProgramSource(1, make_pair(KernelSource.data(), KernelSource.size()));    
    cl::Program Program(Context, ProgramSource);
    
    try
    {
        Program.build(Devices);
    }
    catch(...)
    {
        for(auto aDevice : Devices)
        {
            cout << Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(aDevice) << endl;
        }

		rethrow_exception(std::current_exception());
    }
    
    return Program;
}

cl::CommandQueue ComputePlatform::GetNextCommandQueue()
{
	std::unique_lock<std::mutex> Lock(CommandQueuesMutex);

    // Round Robin Scheduling
    rotate(CommandQueues.begin(), CommandQueues.begin() + 1, CommandQueues.end());
    
    return CommandQueues.back();
}

void ComputePlatform::PrintDeviceName(const std::string& Description, cl::CommandQueue Queue)
{
#if PRINT_DEVICE_NAMES
    auto Device = Queue.getInfo<CL_QUEUE_DEVICE>();
    auto DeviceName = Device.getInfo<CL_DEVICE_NAME>();
    
    cout << Description << ": " << DeviceName << endl;
#endif
}

void ComputePlatform::RecordEvent(const std::vector<std::string>& Tags, cl::Event Event)
{
	if (bProfile)
	{
		auto StartTime = Event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto EndTime = Event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
		auto ElapsedNanoSec = EndTime - StartTime;
		auto ElapsedMilliSec = chrono::duration_cast<chrono::milliseconds>(chrono::nanoseconds(ElapsedNanoSec)).count();
		
		{
			unique_lock<mutex> Lock(RecordEventMutex);
			for (auto Tag : Tags)
			{
				ProfilingMap[Tag] += static_cast<long int>(ElapsedMilliSec);
			}			
		}		
	}	
}

void ComputePlatform::RecordTime(const std::vector<std::string>& Tags, long int Time)
{
	if (bProfile)
	{
		{
			unique_lock<mutex> Lock(RecordEventMutex);
			for (auto Tag : Tags)
			{
				ProfilingMap[Tag] += Time;
			}
		}
	}
}

ostream& operator<< (ostream& out, const ComputePlatform& aComputingPlatform)
{
    auto Platform = aComputingPlatform.Platform;
    
    out << "Platform" << endl;
    out << "\tName   : " << Platform.getInfo<CL_PLATFORM_NAME>() << endl;
    out << "\tVendor : " << Platform.getInfo<CL_PLATFORM_VENDOR>() << endl;
    out << "\tProfile: " << Platform.getInfo<CL_PLATFORM_PROFILE>() << endl;
    out << "\tVersion: " << Platform.getInfo<CL_PLATFORM_VERSION>() << endl;
    
    out << endl;
    
	out << "Devices" << endl;
	for (auto aDevice : aComputingPlatform.Devices)
	{
		out << "\tName           : " << aDevice.getInfo<CL_DEVICE_NAME>() << endl;
		out << "\tVendor         : " << aDevice.getInfo<CL_DEVICE_VENDOR>() << endl;
		out << "\tGlobal Mem Size: " << aDevice.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024) << " Mb" << endl;
		out << "\tLocal Mem Size : " << aDevice.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << " Kb" << endl;

		vector<size_t> MaxWorkGroupSize = aDevice.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
		out << "\tMax Work Item Sizes: ";
		for (auto WorkGroupSize : MaxWorkGroupSize)
		{
			out << WorkGroupSize << " ";
		}
		out << endl;

		out << "\tMax Work Group Size: " << aDevice.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;
		out << "\tMax Compute Units: " << aDevice.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
		out << "\tMem Base Addr Align: " << aDevice.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>() << endl;
        out << endl;
	}    
	out << endl;
    
    return out;
}




