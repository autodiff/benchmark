### INSTALL AD TOOLS ###

FROM ubuntu:20.04 as build_tools

ENV DEBIAN_FRONTEND noninteractive

# General dependencies

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    apt-transport-https \
    autoconf \
    automake \
    ca-certificates \
    clang \
    cmake \
    curl \
    file \
    g++ \
    git \
    libboost-all-dev \
    libeigen3-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libtool \
    llvm-dev \
    make \
    patch \
 && rm -rf /var/lib/apt/lists/*


# Build all tools

COPY tools/ /tools/

RUN cmake \
    -S /tools \
    -B /tools/build \
    -DCMAKE_INSTALL_PREFIX:PATH=/tools-install \
    -DINSTALL_ALL:BOOL=ON \
 && cmake --build tools/build \
 && rm -rf /tools/build

### BUILD BENCHMARKS ###

FROM build_tools as build_benchmarks

COPY benchmarking/ /benchmarking/

RUN cmake \
    -S /benchmarking \
    -B /benchmarking/build \
    -DTOOLS_INSTALL_DIR:PATH=/tools-install \
 && cmake --build /benchmarking/build

### RUN BENCHMARKS ###

FROM ubuntu:20.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    g++ \
    libgoogle-glog-dev \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build_benchmarks /tools-install/ /tools-install/
COPY --from=build_benchmarks /benchmarking/build /benchmarks/

WORKDIR /benchmarks

ENTRYPOINT ["run-parts"]
CMD ["."]
