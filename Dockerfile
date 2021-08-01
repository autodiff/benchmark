ARG ALL_TOOLS=OFF
ARG AUTODIFF_TAG=master
ARG BENCHMARKS=ON
ARG CUSTOM_PROBLEM=OFF

### INSTALL DEPENDENCIES ###

FROM ubuntu:20.04 as builder

ENV DEBIAN_FRONTEND noninteractive

# General dependencies

RUN apt-get update                              \
 && apt-get install -y --no-install-recommends  \
    apt-transport-https                         \
    autoconf                                    \
    automake                                    \
    ca-certificates                             \
    clang                                       \
    cmake                                       \
    curl                                        \
    file                                        \
    g++                                         \
    git                                         \
    libboost-all-dev                            \
    libeigen3-dev                               \
    libgflags-dev                               \
    libgoogle-glog-dev                          \
    libgtest-dev                                \
    libtool                                     \
    llvm-dev                                    \
    make                                        \
    patch                                       \
 && rm -rf /var/lib/apt/lists/*


### INSTALL TOOLS ###

FROM builder as tool_builder

ARG AUTODIFF_TAG
ARG ALL_TOOLS

RUN echo "AUTODIFF_TAG:  ${AUTODIFF_TAG}"       \
 && echo "ALL_TOOLS:     ${ALL_TOOLS}"

COPY tools/ /tools/

RUN cmake                                       \
    -S /tools                                   \
    -B /tools/build                             \
    -DCMAKE_INSTALL_PREFIX:PATH=/tools-install  \
    -DAUTODIFF_TAG:STRING=${AUTODIFF_TAG}       \
    -DALL_TOOLS:BOOL=${ALL_TOOLS}               \
 && cmake --build tools/build                   \
 && rm -rf /tools/build


### BUILD BENCHMARKS ###

FROM tool_builder as benchmark_builder

ARG ALL_TOOLS
ARG BENCHMARKS
ARG CUSTOM_PROBLEM

RUN echo "ALL_TOOLS:      ${ALL_TOOLS}"         \
 && echo "BENCHMARKS:     ${BENCHMARKS}"        \
 && echo "CUSTOM_PROBLEM: ${CUSTOM_PROBLEM}"

COPY benchmarking/ /benchmarking/

RUN cmake                                       \
    -S /benchmarking                            \
    -B /benchmarking/build                      \
    -DTOOLS_INSTALL_DIR:PATH=/tools-install     \
    -DALL_TOOLS:BOOL=${ALL_TOOLS}               \
    -DBENCHMARKS:BOOL=${BENCHMARKS}             \
    -DCUSTOM_PROBLEM:BOOL=${CUSTOM_PROBLEM}     \
 && cmake --build /benchmarking/build -j4


### RUN BENCHMARKS ###

FROM ubuntu:20.04

RUN apt-get update                              \
 && apt-get install -y --no-install-recommends  \
    g++                                         \
    libgoogle-glog-dev                          \
 && rm -rf /var/lib/apt/lists/*

COPY --from=benchmark_builder /tools-install/ /tools-install/
COPY --from=benchmark_builder /benchmarking/build /benchmarks/

WORKDIR /benchmarks

ENTRYPOINT ["run-parts"]
CMD ["."]
