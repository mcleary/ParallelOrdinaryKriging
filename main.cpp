//
//  main.cpp
//  ParallelDTM
//
//  Created by Thales Sabino on 8/25/16.
//  Copyright © 2016 Thales Sabino. All rights reserved.
//

#include <iostream>
#include <algorithm>
#include <numeric>
#include <thread>

#include "CommandLineParser.h"
#include "XYZFile.h"
#include "ComputePlatform.h"
#include "KrigingOperation.h"
#include "KrigingSerial.h"
#include "Timer.h"

using namespace std;

template<class TContainer>
bool BeginsWith(const TContainer& input, const TContainer& match)
{
	return input.size() >= match.size()
		&& equal(match.begin(), match.end(), input.begin());
}

int main(int ArgC, char* ArgV[])
{
	try
	{
#if defined(WIN32) || defined(UNIX)
        unsigned int ConcurentThreadsSupported = std::thread::hardware_concurrency();
		omp_set_num_threads(ConcurentThreadsSupported);
		Eigen::setNbThreads(ConcurentThreadsSupported);
		Eigen::initParallel();
#endif
        
		CommandLineParser CmdParser(ArgC, ArgV);

		if ((!CmdParser.OptionExists("--input") &&
			!CmdParser.OptionExists("--output")) || ArgC < 3)
		{
			cout << "USAGE: " << ArgV[0] << " --input [XYZ File] --output [Output File] {--lags-count [N] --grid-size [Size] --platform [ID] --num-devices [N] --profile --run-serial}" << endl;
			return EXIT_FAILURE;
		}
        
        int PlatformID = 0;
        if(CmdParser.OptionExists("--platform"))
        {
            auto PlatformIDStr = CmdParser.GetOptionValue("--platform");
            PlatformID = std::atoi(PlatformIDStr.data());
        }

		int NumDevices = -1;
		if (CmdParser.OptionExists("--num-devices"))
		{
			auto NumDevicesStr = CmdParser.GetOptionValue("--num-devices");
			NumDevices = std::atoi(NumDevicesStr.data());
		}
        
        int LagsCount = 10;
        if(CmdParser.OptionExists("--lags-count"))
        {
            auto LagsCountStr = CmdParser.GetOptionValue("--lags-count");
            LagsCount = std::atoi(LagsCountStr.data());
        }
        
        int GridSize = 30;
        if(CmdParser.OptionExists("--grid-size"))
        {
            auto GridSizeStr = CmdParser.GetOptionValue("--grid-size");
            GridSize = std::atoi(GridSizeStr.data());
        }
        
        bool bRunSerial = CmdParser.OptionExists("--run-serial");
        bool bProfile = CmdParser.OptionExists("--profile");
        
        auto InputFilepath = CmdParser.GetOptionValue("--input");
        auto OutputFilepath = CmdParser.GetOptionValue("--output");
        
        auto InputPoints = ReadXYZFile(InputFilepath);
        
        int NumberOfPoints = static_cast<int>(InputPoints.size());
        cout << "Number of Points: " << NumberOfPoints << endl;
        
        if(bRunSerial)
        {
            // Run Serial Code
            Serialkriging SerialKrigingOperation;
            
            Timer SerialKrigingTimer;
            
            SerialKrigingOperation.SerialKrigFit(InputPoints, NumberOfPoints, LagsCount);
            
            auto SerialKrigFitElapsed = SerialKrigingTimer.elapsedMilliseconds();
            SerialKrigingTimer = Timer();
            
            auto KrigGrid = SerialKrigingOperation.SerialKrigPred(InputPoints, GridSize);
            
            auto SerialKrigPredElapsed = SerialKrigingTimer.elapsedMilliseconds();
            
            if(bProfile)
            {
                cout << "Profiling Info:" << endl;
                cout << "\t" << "Serial Kriging Fit : " << SerialKrigFitElapsed << " ms" << endl;
                cout << "\t" << "Serial Kriging Pred: " << SerialKrigPredElapsed << " ms" << endl;
                cout << "\t" << "Total: " << SerialKrigFitElapsed + SerialKrigPredElapsed << " ms" << endl;
            }
            
            WriteXYZFile(OutputFilepath, KrigGrid);
        }
        else
        {
            // Run Parallel Code
            ComputePlatform TheComputePlatform(PlatformID, NumDevices);
            cout << TheComputePlatform << endl;
            
            TheComputePlatform.bProfile = bProfile;
            
            KrigingOperation KrigingOperation(TheComputePlatform);
            
            Timer KrigingTimer;
            
            KrigingOperation.KrigFit(InputPoints, NumberOfPoints, LagsCount);
            
            TheComputePlatform.RecordTime({ "TotalKriging", "KrigFit" }, KrigingTimer.elapsedMilliseconds());
            
            KrigingTimer = Timer();
            
            auto KrigGrid = KrigingOperation.KrigPred(InputPoints, GridSize);
            
            TheComputePlatform.RecordTime({ "TotalKriging", "KrigPred" }, KrigingTimer.elapsedMilliseconds());
            
            WriteXYZFile(OutputFilepath, KrigGrid);
            
            if (TheComputePlatform.bProfile)
            {
                long int TotalTime = 0;
                
                cout << "Profiling Info:" << endl;
                for (auto ProfilePair : TheComputePlatform.ProfilingMap)
                {
                    auto EventName = ProfilePair.first;
                    auto EventTime = ProfilePair.second;
                    
                    cout << "\t" << EventName << ": " << EventTime << " ms" << endl;
                    
                    if (!BeginsWith(EventName, string("Total")))
                    {
                        TotalTime += ProfilePair.second;
                    }
                }
                cout << endl;
                cout << "\tTotal: " << TotalTime << " ms" << endl;
            }
        }
	}
	catch (const cl::Error& Exception)
	{		
		cout << "[CL ERROR] " << Exception.what() << " " << Exception.err() << endl;
	}
	catch (const exception& Exception)
	{
		cout << "[ERROR] " << Exception.what() << endl;
	}	

    return EXIT_SUCCESS;
}
