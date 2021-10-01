# Autodifferentiation benchmarks

Scripts and tests to benchmark [`autodiff`](https://autodiff.github.io) against itself and against other C++ autodifferentiation tools.

The objective of this repository is to make it easy to:

* Compare multiple autodifferentiation tools on the benchmark suite 
```zsh
docker build --build-arg ALL_TOOLS=ON -t ad_benchmark_all .
docker run --rm ad_benchmark_all | tee results_all
```
Note that downloading and compiling the autodifferentiation tools, as well as the benchmark suite, may take considerable time.

* Compare two `autodiff` commits/tags/branches on the benchmark suite
```zsh
docker build --build-arg ALL_TOOLS=OFF --build-arg AUTODIFF_TAG=master -t ad_benchmark_master .
docker build --build-arg ALL_TOOLS=OFF --build-arg AUTODIFF_TAG=mybranch -t ad_benchmark_mybranch .

docker run --rm ad_benchmark_master | tee results_master
docker run --rm ad_benchmark_mybranch | tee results_mybranch
```

* Compare multiple autodifferentiation tools on a custom problem
```zsh
# implement a custom problem in benchmarking/custom_problem.cpp, then build it:
docker build --build-arg ALL_TOOLS=ON --build-arg BENCHMARKS=OFF --build-arg CUSTOM_PROBLEM=ON -t ad_custom .
docker run --rm ad_custom | tee results_custom
```

In addition to comparing speed, correctness is also validated by verifying the result against numerical differentiation. If there are errors those are printed to `stderr`.

The output from benchmarks is printed to `stdout` in the format
```
<AD Tool>     <Test name>    <Init time>    <Differentiation time>
```
Here `<Init time>` is the time required to prepare the tool for differentiation of a new function. It is (practically) zero for tools implementing the forward method (numerical, autodiff, adolc, etc), but may be significant for tape-based methods (CppAD, CppAD-CG, etc). Times are average runtimes and are reported in nanoseconds.


Results can be visualized using the included `plotting/visualization.ipynb` notebook.

The benchmarks all involve computing jacobians for vector-valued functions.

* All calculations use `double`s
* For tape methods all available optimizations are employed in recording phase
* Input and outputs are `Eigen` vectors of static or dynamic size
* No multi-threading
* Tools and benchmarks are compiled in `Release` mode (`-O3 -DNDEBUG`)

Please submit a pull request if you have a complementing benchmark problem!


## Usage guide

### Using docker

Docker is the easiest option. The Dockerfile has two arguments: 
 * `AUTODIFF_TAG`: branch/tag/commit of `autodiff` to install, default `master`
 * `ALL_TOOLS`: install other autodiff tools for comparison, default `OFF`
 * `BENCHMARKS`: compile benchmark suite, default `ON`
 * `CUSTOM_PROBLEM`: compile custom problem, default `OFF`

 The Dockerfile is structured in multiple stages so that iterative usage does not require a complete rebuild.

Example:
```zsh
docker build --build-arg ALL_TOOLS=OFF --build-arg AUTODIFF_TAG=d74087d -t ad_benchmark_tag .  # build tagged version of autodiff 
docker build --build-arg ALL_TOOLS=ON --build-arg AUTODIFF_TAG=master -t ad_benchmark_all .    # build all autodiff tools
```
When the docker image is ready the benchmarks are executed as follows:
```zsh
docker run ad_benchmark_tag 
```

### Locally

For development and faster iteration it can be convenient to work outside of docker.
It is recommended to install the tools in a local folder for easy removal later:
```zsh
export BENCHMARK_INSTALL=~/benchmark-install  # modify as desired
```

1. Download, compile, and install desired tools. The `CMakeLists.txt` file has the same arguments as the Dockerfile.
```zsh
mkdir -p tools/build && cd tools/build
cmake ..                                       \
   -DCMAKE_INSTALL_PREFIX=${BENCHMARK_INSTALL} \
   -DAUTODIFF_TAG=master                       \
   -DALL_TOOLS=ON
make
```

2. Then the benchmarks themselves can be compiled.
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
