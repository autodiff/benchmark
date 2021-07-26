# Autodifferentiation benchmarks

Scripts and tests to benchmark `autodiff` against itself and against other C++ autodifferentiation tools. The objective of this repository is to make it easy to:

* Evaluate multiple autodifferentiation tools on a specific problem
* Compare different versions of `autodiff` to track performance

## TODOs

This repository is a work in progress.

* [x] Install tools and build tests in a Dockerfile
* [x] Generate exectuable for each benchmark tool
* [x] Enable dynamically sized tests
* [ ] Write script that runs tests and collects results, include in Dockerfile

## Test requirements

* Input and outputs are Eigen vectors
* Static-sized matrices
* For tape methods use all available optimizations in recording phase

## Running the benchmarks

Docker is the preferred option to compile the different tools:
```
docker build -t benchmarker .
```

## Other autodiff benchmarks

* Adept: https://github.com/rjhogan/Adept-2/tree/master/benchmark
* Ceres: https://github.com/ceres-solver/ceres-solver/tree/master/internal/ceres/autodiff_benchmarks
* "A Benchmark of Selected Algorithmic Differentiation Tools on Some Problems in Computer Vision and Machine Learning"
  - Paper: https://arxiv.org/abs/1807.10129
  - Code: https://github.com/microsoft/ADBench
  - Tools: Adept, ADOL-C, Ceres
* "Automatic Differentiation of Rigid Body Dynamics for Optimal Control and Estimation"
  - Paper: https://arxiv.org/abs/1709.03799
  - Tools: CppAD, CppAD-CG
