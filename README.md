# Parallel Ordinary Kriging using OpenCL and OpenMP

## Description
This is an implementation of the Ordinary Kriging algorithm using parallel technologies.

## Compiling
We provide a **CMake** script in order to make compiling the code straightforward. You just need a C++11 compiler with OpenMP support and the OpenCL SDK installed. We will provide detailed information in the future.

## Running
After compiling the code you should see `ParallelOK` executable in the build folder. To use it simple do

```bash
ParallelOK --input [XYZ File] --output [Output File] --lags-count [N] --grid-size [N]
```

- `--lags-count [N]`: The number of lags when generated the empirical semivariogram.
- `--grid-size [N]`: Creates a *NxN* grid to make predictions.

### Optional Arguments
- `--profile`: Will print detailed information about steps runtimes
- `--platform [ID]`: Select the OpenCL platform with ID to run OpenCL
- `--num-devices [N]`: Number of devices to use, omit to use all available devices.
- `--run-serial`: If present will run a serial version of the Ordinary Kriging. This option forces the program to run in serial mode even if `--platform` was provided.

## XYZ File
The XYZ File is a simple point cloud format where each line represents a point in 3D space. For the kriging algorithm, each *z* value is considered a response value for a random variable at location *(x,y)*.

### Example of a XYZ file
```
-113.24195 40.01692 -2.07465272
-71.99837 38.22831 -0.58196695
-81.89808 28.22333 -1.54724761
-85.5524 49.05601 0.99257163
```

## Future Work
- Structure the code as a library to incorporate in other applications
- Review the matrix-vector multiplication code for GPUs

## Contact
Please, feel free to file an issue directly in this repository or send an email to [tluis@ice.ufjf.br](mailto:tluis@ice.ufjf.br)

## Implementation Results

To show that our implementation can scale across multiple compute devices we show here two graphs. The first is the amount of time spent by the prediction step with the increase in the number of compute devices. The second graph is showing that our implementation can achieve a quasi-linear speedup with the increase in the number of devices being used. 

Our results were published in the [CCIS 2016](http://www.epacis.net/ccis2016/en/index.php). The link to the full paper can be found below.

### [A parallel implementation of the ordinary kriging algorithm for heterogeneous computing environments](https://www.dropbox.com/s/d39y6495u6zkvt3/parallel-implementation-ordinary.pdf?dl=0)

![Prediction Runtimes](https://bytebucket.org/pgmc-ufjf/parallelordinarykriging/raw/224d7787c8963d70c17501b42cb80c2a692c1855/Figures/PredRuntime.png)

![Prediction Speedups](https://bytebucket.org/pgmc-ufjf/parallelordinarykriging/raw/224d7787c8963d70c17501b42cb80c2a692c1855/Figures/PredSpeedup.png)

