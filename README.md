# Autodifferentiation benchmarks

Scripts and tests to benchmark `autodiff` against itself and against other C++ autodifferentiation tools. The objective of this repository is to make it easy to:

* Compare performance of multiple autodifferentiation tools 
```
docker build --build-arg ALL_TOOLS=ON --build-arg AUTODIFF_TAG=master -t ad_benchmark_all .
docker run --rm ad_benchmark_all | tee results_all
```

* Compare performance of two `autodiff` branches
```
docker build --build-arg ALL_TOOLS=OFF --build-arg AUTODIFF_TAG=master -t ad_benchmark_master .      # build tagged version of autodiff 
docker build --build-arg ALL_TOOLS=OFF --build-arg AUTODIFF_TAG=mybranch -t ad_benchmark_mybranch .  # build tagged version of autodiff 

docker run --rm ad_benchmark_master | tee results_master
docker run --rm ad_benchmark_mybranch | tee results_mybranch
```

* Evaluate autodifferentiation tools on a custom problem: TODO

Results can be visualized using the `plotting/visualization.ipynb` notebook.

The benchmarks all involve computing jacobians for vector-valued functions.
* Input and outputs are Eigen vectors of static or dynamic size
* All calculations use `double`s
* Tools and benchmarks are compiled in `Release` mode (`-O3 -DNDEBUG`)
* For tape methods all available optimizations are employed in recording phase

Please submit a pull request if you have a benchmark that would complement the included ones.


## Usage guide

### Using docker

Docker is the easiest option. The Dockerfile has two arguments: 
 * `AUTODIFF_TAG`: install specific version of `autodiff`, default `master`
 * `ALL_TOOLS`: install other autodiff tools for comparison, default `OFF`

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


## Other resources

Links to previous autodifferentiation benchmarks:

* Adept: https://github.com/rjhogan/Adept-2/tree/master/benchmark
* Ceres: https://github.com/ceres-solver/ceres-solver/tree/master/internal/ceres/autodiff_benchmarks
* "A Benchmark of Selected Algorithmic Differentiation Tools on Some Problems in Computer Vision and Machine Learning"
  - Paper: https://arxiv.org/abs/1807.10129
  - Code: https://github.com/microsoft/ADBench
  - Tools: Adept, ADOL-C, Ceres
* "Automatic Differentiation of Rigid Body Dynamics for Optimal Control and Estimation"
  - Paper: https://arxiv.org/abs/1709.03799
  - Tools: CppAD, CppAD-CG
