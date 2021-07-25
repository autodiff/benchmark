cmake_minimum_required(VERSION 3.15)
project(ad_benchmark_tools)

option(INSTALL_ALL      "Install all autodiff tools" OFF)
option(INSTALL_ADEPT    "Install adept"              OFF)
option(INSTALL_ADOLC    "Install adolc"              OFF)
option(INSTALL_AUTODIFF "Install autodiff"           OFF)
option(INSTALL_CERES    "Install ceres"              OFF)
option(INSTALL_CPPAD    "Install cppad"              OFF)
option(INSTALL_CPPADCG  "Install cppad-cg"           OFF)
option(INSTALL_SACADO   "Install sacado"             OFF)

set(CMAKE_BUILD_TYPE Release)

include(ExternalProject)

file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX})

if(${INSTALL_CPPADCG} AND NOT ${INSTALL_CPPAD})
  message(WARNING "Setting INSTALL_CPPAD=ON since it is required by cppad-cg")
  set(INSTALL_CPPAD ON)
endif()

if(${INSTALL_ADEPT} OR ${INSTALL_ALL})
  # Compiled
  externalproject_add(
      adept-external
      GIT_REPOSITORY       https://github.com/rjhogan/Adept-2.git
      GIT_TAG              23b5bc07accb3c3289b062eff9f188aa2002fa6f
      GIT_SHALLOW          ON
      GIT_CONFIG           advice.detachedHead=false
      # Need this patch to make adept work with Eigen (prevents gobbling constructor & ambiguous operator*)
      PATCH_COMMAND        patch -p0 -ruN -d <SOURCE_DIR> < ${CMAKE_CURRENT_SOURCE_DIR}/patches/adept_sfinae.patch
      CONFIGURE_COMMAND    autoreconf -fi <SOURCE_DIR>
      COMMAND              ${CMAKE_COMMAND} -E env
                           CC=${CMAKE_C_COMPILER}
                           CFLAGS=${CMAKE_C_FLAGS_RELEASE}
                           CXX=${CMAKE_CXX_COMPILER}
                           CXXFLAGS=${CMAKE_CXX_FLAGS_RELEASE}
                           <SOURCE_DIR>/configure
                           --prefix=${CMAKE_INSTALL_PREFIX}
                           --disable-openmp
      BUILD_IN_SOURCE      ON
  )
  file(TOUCH ${CMAKE_INSTALL_PREFIX}/AD_BENCHMARK_ADEPT)
endif()

if(${INSTALL_ADOLC} OR ${INSTALL_ALL})
  # Compiled
  externalproject_add(
      adolc-external
      GIT_REPOSITORY       https://github.com/coin-or/ADOL-C.git
      GIT_TAG              releases/2.7.2
      GIT_SHALLOW          ON
      GIT_CONFIG           advice.detachedHead=false
      CONFIGURE_COMMAND    autoreconf -fi <SOURCE_DIR>
      COMMAND              ${CMAKE_COMMAND} -E env
                           CC=${CMAKE_C_COMPILER}
                           CFLAGS=${CMAKE_C_FLAGS_RELEASE}
                           CXX=${CMAKE_CXX_COMPILER}
                           CXXFLAGS=${CMAKE_CXX_FLAGS_RELEASE}
                           <SOURCE_DIR>/configure
                           --prefix=${CMAKE_INSTALL_PREFIX}
                           --includedir=${CMAKE_INSTALL_PREFIX}/include
                           --libdir=${CMAKE_INSTALL_PREFIX}/lib
                           --with-cxxflags=${CMAKE_CXX_FLAGS_RELEASE}
                           --enable-traceless-refcounting  # hide warning in tapeless mode
      BUILD_IN_SOURCE      ON
  )
  file(TOUCH ${CMAKE_INSTALL_PREFIX}/AD_BENCHMARK_ADOLC)
endif()

if(${INSTALL_AUTODIFF} OR ${INSTALL_ALL})
  # Header-only
  externalproject_add(
      autodiff-external
      GIT_REPOSITORY       https://github.com/autodiff/autodiff.git
      GIT_TAG              v0.6.1
      GIT_SHALLOW          ON
      GIT_CONFIG           advice.detachedHead=false
      CMAKE_ARGS           -DAUTODIFF_BUILD_TESTS:BOOL=OFF
                           -DAUTODIFF_BUILD_EXAMPLES:BOOL=OFF
                           -DAUTODIFF_BUILD_PYTHON:BOOL=OFF
                           -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  )
  file(TOUCH ${CMAKE_INSTALL_PREFIX}/AD_BENCHMARK_AUTODIFF)
endif()

if(${INSTALL_CERES} OR ${INSTALL_ALL})
  # Compiled
  externalproject_add(
      ceres-external
      GIT_REPOSITORY       https://github.com/ceres-solver/ceres-solver.git
      GIT_TAG              2.0.0
      GIT_SHALLOW          ON
      GIT_CONFIG           advice.detachedHead=false
      CMAKE_ARGS           -DCMAKE_BUILD_TYPE=Release
                           -DBUILD_TESTING:BOOL=OFF
                           -DBUILD_EXAMPLES:BOOL=OFF
                           -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
  )
  file(TOUCH ${CMAKE_INSTALL_PREFIX}/AD_BENCHMARK_CERES)
endif()

if(${INSTALL_CPPAD} OR ${INSTALL_ALL})
  # Compiled
  externalproject_add(
      cppad-external
      GIT_REPOSITORY       https://github.com/coin-or/CppAD.git
      GIT_TAG              20210000.6
      GIT_SHALLOW          ON
      GIT_CONFIG           advice.detachedHead=false
      CMAKE_ARGS           -Dcppad_cxx_flags=${CMAKE_CXX_FLAGS_RELEASE}
                           -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
  )
  externalproject_add_steptargets(cppad-external install)
  file(TOUCH ${CMAKE_INSTALL_PREFIX}/AD_BENCHMARK_CPPAD)
endif()

if(${INSTALL_CPPADCG} OR ${INSTALL_ALL})
  # Header-only
  externalproject_add(
      cppad-cg-external
      GIT_REPOSITORY       https://github.com/joaoleal/CppADCodeGen.git
      GIT_TAG              v2.4.3
      GIT_SHALLOW          ON
      GIT_CONFIG           advice.detachedHead=false
      CMAKE_ARGS           -DCPPAD_HOME=${CMAKE_INSTALL_PREFIX}/include
                           -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
      DEPENDS              cppad-external-install
  )
  file(TOUCH ${CMAKE_INSTALL_PREFIX}/AD_BENCHMARK_CPPADCG)
endif()

if(${INSTALL_SACADO} OR ${INSTALL_ALL})
  # Compiled
  externalproject_add(
      sacado-external
      GIT_REPOSITORY       https://github.com/trilinos/Trilinos.git
      GIT_TAG              trilinos-release-13-0-1
      GIT_SHALLOW          ON
      GIT_CONFIG           advice.detachedHead=false
      CMAKE_ARGS           -DCMAKE_BUILD_TYPE=Release
                           -DTrilinos_ENABLE_Sacado:BOOL=ON
                           -DTrilinos_ENABLE_Fortran:BOOL=OFF
                           -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF
                           -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
  )
  file(TOUCH ${CMAKE_INSTALL_PREFIX}/AD_BENCHMARK_SACADO)
endif()

