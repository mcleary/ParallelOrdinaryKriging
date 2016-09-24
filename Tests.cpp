
#include "CommandLineParser.h"
#include "ComputePlatform.h"
#include "LinearAlgebraOperation.h"
#include "FillBufferOperation.h"

#include <iostream>
#include <vector>

#include <omp.h>

using namespace std;

int main(int Argc, char* ArgV[])
{
	try
	{
		CommandLineParser CmdParser{ Argc, ArgV };

		int PlatformID = 0;
		if (CmdParser.OptionExists("--platform"))
		{
			auto PlatformIDStr = CmdParser.GetOptionValue("--platform");
			PlatformID = std::atoi(PlatformIDStr.data());
		}

		int TestToPerform = 0;
		if (CmdParser.OptionExists("--test"))
		{
			auto TestIDStr = CmdParser.GetOptionValue("--test");
			TestToPerform = std::atoi(TestIDStr.data());
		}

		ComputePlatform TheComputePlatform(PlatformID);
		cout << TheComputePlatform << endl;

		if (TestToPerform == 0)
		{
			auto Queue = TheComputePlatform.GetNextCommandQueue();

			int N = 10;
			int BufferSize = N * sizeof(float);
			cl::Buffer aBuffer(TheComputePlatform.Context, CL_MEM_READ_WRITE, BufferSize);

			const float One = 1.0f;
			cl::Event FillBufferEvent;
			Queue.enqueueFillBuffer(aBuffer, One, 0, BufferSize, nullptr, &FillBufferEvent);
			FillBufferEvent.wait();

			vector<float> BufferContents(N);
			Queue.enqueueReadBuffer(aBuffer, CL_TRUE, 0, BufferSize, BufferContents.data());

			for (auto i : BufferContents)
			{
				cout << i << " ";
			}
			cout << endl;
		}

		if (TestToPerform == 1)
		{
			auto Queue = TheComputePlatform.GetNextCommandQueue();

			cl::Buffer aBuffer(TheComputePlatform.Context, CL_MEM_READ_WRITE, sizeof(int));

			int Value;
			Queue.enqueueReadBuffer(aBuffer, CL_TRUE, 0, sizeof(int), &Value);

			cout << Value << endl;
		}

		if (TestToPerform == 2)
		{
			int NumDevices = static_cast<int>(TheComputePlatform.Devices.size());
			int NumThreads = 2;

			LinearAlgebraOperation LinAlg{ TheComputePlatform };
			FillBufferOperation FillBuffer{ TheComputePlatform };

			int N = 10000;
			const int GridSize = 30;
			int MatrixSize = N * N * sizeof(double);
			int VectorSize = N * sizeof(double);

			vector<double> ResultVector(GridSize * GridSize);

#		pragma omp parallel num_threads(NumThreads)
			{
				int ThreadId = omp_get_thread_num();

#			pragma omp critical
				cout << "Thread ID: " << ThreadId << endl;

				//auto Queue = TheComputePlatform.GetNextCommandQueue();
				int DeviceID = ThreadId % NumDevices;
				cl::CommandQueue Queue(TheComputePlatform.Context, TheComputePlatform.Devices[DeviceID], CL_QUEUE_PROFILING_ENABLE);

				cl::Buffer MatrixBuffer(TheComputePlatform.Context, CL_MEM_READ_WRITE, MatrixSize);
				cl::Buffer VectorBuffer(TheComputePlatform.Context, CL_MEM_READ_WRITE, VectorSize);
				cl::Buffer ResultBuffer(TheComputePlatform.Context, CL_MEM_READ_WRITE, VectorSize);
				cl::Buffer CacheBuffer(TheComputePlatform.Context, CL_MEM_READ_WRITE, VectorSize);

				FillBuffer.FillDoubleBuffer(Queue, MatrixBuffer, 1.0, N * N).wait();
				FillBuffer.FillDoubleBuffer(Queue, VectorBuffer, 1.0, N).wait();

#			pragma omp for
				for (int i = 0; i < GridSize; ++i)
				{
#				pragma omp critical
					cout << i << " " << flush;

					for (int j = 0; j < GridSize; ++j)
					{
						LinAlg.MatVecMul(Queue, MatrixBuffer, VectorBuffer, ResultBuffer, N).wait();
						double DotResult = LinAlg.DotProduct(Queue, VectorBuffer, ResultBuffer, N, CacheBuffer);

						ResultVector[i + j * GridSize] = DotResult;
					}
				}
			}
		}
	}
	catch (cl::Error& err)
	{
		cout << "CL ERROR: " << err.what() << " " << err.err() << endl;
	}
	catch (runtime_error& e)
	{
		cout << "ERROR: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Unknown Error" << endl;
	}

	return 0;
}
