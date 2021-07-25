# Autodifferentiation benchmarks

Things that should be easy:
 - Run benchmarks for specific `autodiff` version
 - Run benchmarks for all tools
 - Update results (binder)

TODOs

* [x] Generate benchmark executable for each tool
* [ ] Make large tests dynamically sized
* [ ] Executable generates data file tagged with name
* [ ] Plotting tool reads all datafiles

## Scope

* Dense jacobians in Eigen functions
* No branching
* Static-sized matrices
* For tape methods use all available optimizations in recording phase

## Running the benchmarks

Docker is the preferred option to compile the different tools:
```
docker build -t benchmarker .
```

## Other benchmarks

* Adept benchmarks: https://github.com/rjhogan/Adept-2/tree/master/benchmark
* Ceres benchmarks: https://github.com/ceres-solver/ceres-solver/tree/master/internal/ceres/autodiff_benchmarks
* "A Benchmark of Selected Algorithmic Differentiation Tools on Some Problems in Computer Vision and Machine Learning"
  - Paper: https://arxiv.org/abs/1807.10129
  - Code: https://github.com/microsoft/ADBench
  - Tools: Adept, ADOL-C, Ceres
* "Automatic Differentiation of Rigid Body Dynamics for Optimal Control and Estimation"
  - Paper: https://arxiv.org/abs/1709.03799
  - Tools: CppAD, CppAD-CG
