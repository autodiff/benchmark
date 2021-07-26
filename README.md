# Autodifferentiation benchmarks

Scripts and tests to benchmark `autodiff` against itself and against other C++ autodifferentiation tools. The objective of this repository is to make it easy to:

* Evaluate multiple autodifferentiation tools on a specific problem
* Compare different versions of `autodiff` to track performance

## Test requirements

* Input and outputs are Eigen vectors of static or dynamic size
* All calculations are in doubles
* All tools and benchmarks are compiled in `Release` mode (`-O3 -DNDEBUG`)
* For tape methods use all available optimizations in recording phase

## Building and running the benchmarks

### Using docker

Docker is probably the easiest option.

1. Build the Dockerfile

```zsh
docker build -t benchmarks .
```

2. Run the benchmarks and collect results
```zsh
docker run benchmarks | tee plotting/data
```

3. Visualize the results with the `plotting/visualization.ipynb` notebook.

### Locally

For development and faster iteration it can be convenient to work outside of docker.
It is recommended to install the tools in a local folder for easy removal:
```zsh
export BENCHMARK_INSTALL=~/benchmark-install  # modify as desired
```

1. Download, compile, and install desired tools
```zsh
mkdir -p tools/build && cd tools/build
cmake ..                                       \
   -DCMAKE_INSTALL_PREFIX=${BENCHMARK_INSTALL} \
   -DINSTALL_ALL=ON     # or select individual tools
make
```

2. Now the benchmarks themselves can be compiled.
```zsh
mkdir -p benchmarking/build && cd benchmarking/build
cmake ..                                   \
  -DTOOLS_INSTALL_DIR=${BENCHMARK_INSTALL}
make
```

3. Now it should be possible to run the different benchmarks. E.g.:
```zsh
./bm_autodiff_real
```


## Other autodifferentiation benchmarks

* Adept: https://github.com/rjhogan/Adept-2/tree/master/benchmark
* Ceres: https://github.com/ceres-solver/ceres-solver/tree/master/internal/ceres/autodiff_benchmarks
* "A Benchmark of Selected Algorithmic Differentiation Tools on Some Problems in Computer Vision and Machine Learning"
  - Paper: https://arxiv.org/abs/1807.10129
  - Code: https://github.com/microsoft/ADBench
  - Tools: Adept, ADOL-C, Ceres
* "Automatic Differentiation of Rigid Body Dynamics for Optimal Control and Estimation"
  - Paper: https://arxiv.org/abs/1709.03799
  - Tools: CppAD, CppAD-CG
