ARG AUTODIFF_TAG=master
ARG ALL_TOOLS=OFF

### INSTALL DEPENDENCIES ###

FROM ubuntu:20.04 as builder

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

### INSTALL TOOLS ###

FROM builder as tool_builder

COPY tools/ /tools/

ARG AUTODIFF_TAG
ARG ALL_TOOLS

RUN echo "AUTODIFF_TAG:  ${AUTODIFF_TAG}" \
 && echo "ALL_TOOLS:     ${ALL_TOOLS}"

RUN cmake \
    -S /tools \
    -B /tools/build \
    -DCMAKE_INSTALL_PREFIX:PATH=/tools-install \
    -DAUTODIFF_TAG:STRING=${AUTODIFF_TAG} \
    -DALL_TOOLS:BOOL=${ALL_TOOLS} \
 && cmake --build tools/build \
 && rm -rf /tools/build

### BUILD BENCHMARKS ###

FROM tool_builder as benchmark_builder

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

COPY --from=benchmark_builder /tools-install/ /tools-install/
COPY --from=benchmark_builder /benchmarking/build /benchmarks/

WORKDIR /benchmarks

ENTRYPOINT ["run-parts"]
CMD ["."]
