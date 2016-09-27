
#include "KrigingOperation.h"
#include "KrigingCommon.h"
#include "ReductionOperation.h"
#include "DistancesMatrixOperation.h"
#include "FillBufferOperation.h"
#include "LinearAlgebraOperation.h"
#include "Timer.h"

#include <iostream>

using namespace std;

KrigingOperation::KrigingOperation(ComputePlatform& Platform) :
    ThePlatform(Platform)
{
    KrigingProgram = ThePlatform.CreateProgram("kernels/Kriging.cl");
}

void KrigingOperation::KrigFit(const PointVector& InputPoints, int NumberOfPoints, int LagsCount)
{
	this->NumberOfPoints = NumberOfPoints;

	auto Queue = ThePlatform.GetNextCommandQueue();

	DEBUG_OPERATION;

	cl::Buffer PointsBuffer(ThePlatform.Context, CL_MEM_READ_ONLY, NumberOfPoints * sizeof(PointXYZ));
	Queue.enqueueWriteBuffer(PointsBuffer, CL_TRUE, 0, NumberOfPoints * sizeof(PointXYZ), InputPoints.data());

	ReductionOperation ReductionOperation{ ThePlatform };
	DistancesMatrixOperation DistancesMatrixOperation{ ThePlatform };
	FillBufferOperation FillBufferOperation{ ThePlatform };

	MinPoint = ReductionOperation.ReducePoints(PointsBuffer, NumberOfPoints, ReductionOp::Min);
	MaxPoint = ReductionOperation.ReducePoints(PointsBuffer, NumberOfPoints, ReductionOp::Max);

	cout << "MinPoint: " << MinPoint << endl;
	cout << "MaxPoint: " << MaxPoint << endl;

	const float Cutoff = Dist(MaxPoint.x, MaxPoint.y, MinPoint.x, MinPoint.y) / 3.0f;
	auto LagRanges = GetLagRanges(Cutoff, LagsCount);

	const int DistancesMatrixElementCount = NumberOfPoints * NumberOfPoints;
	const int DistancesMatrixBufferSize = DistancesMatrixElementCount * sizeof(float);
	cl::Buffer DistancesMatrixBuffer(ThePlatform.Context, CL_MEM_READ_WRITE, DistancesMatrixBufferSize);

	cout << "Computing Distances Matrix ... " << flush;
	auto ComputeDistMatrixEvent = DistancesMatrixOperation.ComputeMatrix(PointsBuffer, NumberOfPoints, DistancesMatrixBuffer);
	if (ThePlatform.bProfile)
	{
		ComputeDistMatrixEvent.wait();
		ThePlatform.RecordEvent({ "DistancesMatrix" }, ComputeDistMatrixEvent);
	}
	cout << "done" << endl;

	auto SemivariogramKernel = cl::make_kernel<
		cl::Buffer,
		cl::Buffer,
		int,
		float,
		float,
		cl::Buffer,
		cl::Buffer,
		cl::Buffer>
		(KrigingProgram, "SemivariogramKernel");	

	cout << "Computing Semivariogram ... " << flush;

	vector<float> EmpiricalSemivariogramX(LagsCount, std::numeric_limits<float>::infinity());
	vector<float> EmpiricalSemivariogramY(LagsCount, std::numeric_limits<float>::infinity());

    Timer SemivariogramTimer;

#	pragma omp parallel num_threads(static_cast<int>(ThePlatform.Devices.size()))
	{
		auto SemivarQueue = ThePlatform.GetNextCommandQueue();		

		cl::Buffer LocalPointsBuffer;
		cl::Buffer LocalDistancesMatrixBuffer;

		if (SemivarQueue() != Queue())
		{
			LocalPointsBuffer = cl::Buffer(ThePlatform.Context, CL_MEM_READ_ONLY, NumberOfPoints * sizeof(PointXYZ));
			LocalDistancesMatrixBuffer = cl::Buffer(ThePlatform.Context, CL_MEM_READ_ONLY, DistancesMatrixBufferSize);
			
			SemivarQueue.enqueueCopyBuffer(PointsBuffer, LocalPointsBuffer, 0, 0, NumberOfPoints * sizeof(PointXYZ));
			SemivarQueue.enqueueCopyBuffer(DistancesMatrixBuffer, LocalDistancesMatrixBuffer, 0, 0, DistancesMatrixBufferSize);
		}
		else
		{
			LocalPointsBuffer = PointsBuffer;
			LocalDistancesMatrixBuffer = DistancesMatrixBuffer;
		}

		cl::Buffer ValidValuesCountBuffer(ThePlatform.Context, CL_MEM_WRITE_ONLY, sizeof(int));
		cl::Buffer DistancesValuesBuffer(ThePlatform.Context, CL_MEM_READ_WRITE, DistancesMatrixBufferSize);
		cl::Buffer SemivarValuesBuffer(ThePlatform.Context, CL_MEM_READ_WRITE, DistancesMatrixBufferSize);

#		pragma omp for
		for (int LagIndex = 0; LagIndex < LagsCount; ++LagIndex)
		{
			auto FillEvent1 = FillBufferOperation.FillFloatBuffer(SemivarQueue, DistancesValuesBuffer, 0.0f, DistancesMatrixElementCount);
			auto FillEvent2 = FillBufferOperation.FillFloatBuffer(SemivarQueue, SemivarValuesBuffer, 0.0f, DistancesMatrixElementCount);
			auto FillEvent3 = FillBufferOperation.FillIntBuffer(SemivarQueue, ValidValuesCountBuffer, 0, 1);

			vector<cl::Event> FillBufferEvents = { FillEvent1, FillEvent2, FillEvent3 };

			const float RangeMin = LagRanges[LagIndex * 2 + 0];
			const float RangeMax = LagRanges[LagIndex * 2 + 1];

			auto SemivarKernelEvent = SemivariogramKernel(
				cl::EnqueueArgs(SemivarQueue, FillBufferEvents, cl::NDRange(NumberOfPoints)),
				LocalPointsBuffer,
				LocalDistancesMatrixBuffer,
				NumberOfPoints,
				RangeMin,
				RangeMax,
				DistancesValuesBuffer,
				SemivarValuesBuffer,
				ValidValuesCountBuffer);

			int ValidValuesCount;
			SemivarQueue.enqueueReadBuffer(ValidValuesCountBuffer, CL_TRUE, 0, sizeof(int), &ValidValuesCount);

			if (ValidValuesCount > 0)
			{
				float AvgDistance = ReductionOperation.Reduce(SemivarQueue, DistancesValuesBuffer, NumberOfPoints * NumberOfPoints, ReductionOp::Sum);
				float AvgSemivar = ReductionOperation.Reduce(SemivarQueue, SemivarValuesBuffer, NumberOfPoints * NumberOfPoints, ReductionOp::Sum);

				AvgDistance /= ValidValuesCount;
				AvgSemivar /= ValidValuesCount;

				EmpiricalSemivariogramX[LagIndex] = AvgDistance;
				EmpiricalSemivariogramY[LagIndex] = 0.5f * AvgSemivar;
			}
		}
	}
    
    ThePlatform.RecordTime({ "Semivariogram" }, SemivariogramTimer.elapsedMilliseconds());

	cout << "done" << endl;

	// Remove Invalid Elements
	auto IsInfPred = [](float Element) { return isinf(Element); };
	EmpiricalSemivariogramX.erase(remove_if(EmpiricalSemivariogramX.begin(), EmpiricalSemivariogramX.end(), IsInfPred), EmpiricalSemivariogramX.end());
	EmpiricalSemivariogramY.erase(remove_if(EmpiricalSemivariogramY.begin(), EmpiricalSemivariogramY.end(), IsInfPred), EmpiricalSemivariogramY.end());

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
	const int CovarianceMatrixBufferCount = (NumberOfPoints + 1) * (NumberOfPoints + 1);
	const int CovarianceMatrixBufferSize = CovarianceMatrixBufferCount * sizeof(float);
	cl::Buffer CovarianceMatrixBuffer(ThePlatform.Context, CL_MEM_READ_WRITE, CovarianceMatrixBufferSize);

	// Cria a matriz de covari�ncia com preenchida com 1's e um �nico zero no �ltimo elemento
	auto CovMatrixFillBufferEvent = FillBufferOperation.FillFloatBuffer(CovarianceMatrixBuffer, 1.0f, CovarianceMatrixBufferCount);

	auto CovMatrixKernel = cl::make_kernel<
		cl::Buffer,
		cl::Buffer,
		int,
		float,
		float,
		float>
		(KrigingProgram, "CovarianceMatrixKernel");

	auto CovMatrixKernelEvent = CovMatrixKernel(
		cl::EnqueueArgs(Queue, CovMatrixFillBufferEvent, cl::NDRange(NumberOfPoints)),
		DistancesMatrixBuffer,
		CovarianceMatrixBuffer,
		NumberOfPoints,
		Nugget,
		Range,
		Sill
	);

	Eigen::MatrixXf CovMatrix(NumberOfPoints + 1, NumberOfPoints + 1);
	Queue.enqueueReadBuffer(CovarianceMatrixBuffer, CL_TRUE, 0, CovarianceMatrixBufferSize, CovMatrix.data());

	ThePlatform.RecordEvent({ "CovarianceMatrix" }, CovMatrixKernelEvent);

	CovMatrix(NumberOfPoints, NumberOfPoints) = 0.0f;
	cout << "done" << endl;

	Timer InvertingMatrixTimer;

	cout << "Inverting Covariance Matrix ..." << flush;
	InvCovMatrix = CovMatrix.cast<double>();
	InvCovMatrix = InvCovMatrix.inverse();
	cout << "done" << endl;

	ThePlatform.RecordTime({ "InverseMatrix" }, InvertingMatrixTimer.elapsedMilliseconds());
}

vector<PointXYZ> KrigingOperation::KrigPred(const PointVector& InputPoints, int GridSize)
{
	cout << "Predicting ... " << flush;	

    LinearAlgebraOperation LinAlgOperation{ ThePlatform };
    FillBufferOperation FillBufferOperation{ ThePlatform };

	vector<PointXYZ> Grid(GridSize * GridSize);
	float GridDeltaX = (MaxPoint.x - MinPoint.x) / GridSize;
	float GridDeltaY = (MaxPoint.y - MinPoint.y) / GridSize;

    vector<double> ZValues(NumberOfPoints + 1);
    for(int i = 0; i < NumberOfPoints; ++i)
    {
        ZValues[i] = InputPoints[i].z;
    }
    ZValues[NumberOfPoints] = 1.0;
    
	auto PredicionCovarianceKernel = cl::make_kernel<
		cl::Buffer,
		cl::Buffer,
		double,
		double,
		double,
		double,
		double>(KrigingProgram, "PredictionCovariance");	

	const int CovMatrixRowsCount = NumberOfPoints + 1;
	const int PredBuffersSize = CovMatrixRowsCount * sizeof(double);
	const int InvCovMatrixSize = CovMatrixRowsCount * CovMatrixRowsCount * sizeof(double);

#	pragma omp parallel num_threads(static_cast<int>(ThePlatform.Devices.size()))
	{
		auto Queue = ThePlatform.GetNextCommandQueue();

		cl::Buffer PointsBuffer(ThePlatform.Context, CL_MEM_READ_ONLY, NumberOfPoints * sizeof(PointXYZ));
		cl::Buffer InvCovMatrixBuffer(ThePlatform.Context, CL_MEM_READ_ONLY, InvCovMatrixSize);
		cl::Buffer ZValuesBuffer(ThePlatform.Context, CL_MEM_READ_ONLY, PredBuffersSize);
		cl::Buffer InvAXRBuffer(ThePlatform.Context, CL_MEM_READ_WRITE, PredBuffersSize);
		cl::Buffer RBuffer(ThePlatform.Context, CL_MEM_READ_WRITE, PredBuffersSize);
		cl::Buffer Cache(ThePlatform.Context, CL_MEM_READ_WRITE, CovMatrixRowsCount * sizeof(double));

		cl::Event WriteMatrixEvent;
		cl::Event WriteZBufferEvent;
		cl::Event WritePointsEvent;

		Queue.enqueueWriteBuffer(PointsBuffer, CL_FALSE, 0, NumberOfPoints * sizeof(PointXYZ), InputPoints.data(), nullptr, &WritePointsEvent);
		Queue.enqueueWriteBuffer(InvCovMatrixBuffer, CL_FALSE, 0, InvCovMatrixSize, InvCovMatrix.data(), nullptr, &WriteMatrixEvent);
		Queue.enqueueWriteBuffer(ZValuesBuffer, CL_FALSE, 0, PredBuffersSize, ZValues.data(), nullptr, &WriteZBufferEvent);
		auto FillRBufferEvent = FillBufferOperation.FillDoubleBuffer(Queue, RBuffer, 1.0, CovMatrixRowsCount);

		cl::WaitForEvents({ WriteMatrixEvent, WriteZBufferEvent, FillRBufferEvent, WritePointsEvent });

#		pragma omp for
		for (int i = 0; i < GridSize; ++i)
		{
#			pragma omp critical
			cout << i << " " << flush;

			for (int j = 0; j < GridSize; ++j)
			{
				float GridX = MinPoint.x + i * GridDeltaX;
				float GridY = MinPoint.y + j * GridDeltaY;

				PredicionCovarianceKernel(cl::EnqueueArgs(Queue, cl::NDRange(NumberOfPoints)),
					PointsBuffer,
					RBuffer,
					GridX,
					GridY,
					Nugget,
					Range,
					Sill);

				auto MatVecMulEvent = LinAlgOperation.MatVecMul(Queue, InvCovMatrixBuffer, RBuffer, InvAXRBuffer, CovMatrixRowsCount);
				double GridZ = LinAlgOperation.DotProduct(Queue, InvAXRBuffer, ZValuesBuffer, CovMatrixRowsCount, Cache);

				Grid[i + j * GridSize] = PointXYZ(GridX, GridY, GridZ);
			}
		}
	}     

	cout << "done" << endl;

	return Grid;
}
