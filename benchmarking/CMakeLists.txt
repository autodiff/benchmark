cmake_minimum_required(VERSION 3.15)
project(ad_benchmark)

find_package(Eigen3 REQUIRED)

set(TOOLS_INSTALL_DIR "/usr" CACHE PATH "Tool installation directory")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# numerical benchmark
add_executable(bm_numerical main.cpp)
target_include_directories(bm_numerical PRIVATE ${TOOLS_INSTALL_DIR}/include)
target_link_directories(bm_numerical PRIVATE ${TOOLS_INSTALL_DIR}/lib)
target_link_libraries(bm_numerical Eigen3::Eigen pthread)
target_compile_definitions(bm_numerical PRIVATE ENABLE_NUMERICAL)

# autodiff benchmark
find_package(autodiff QUIET PATHS ${TOOLS_INSTALL_DIR}/lib/cmake)
if(${autodiff_FOUND})
  add_executable(bm_autodiff_dual main.cpp)
  target_include_directories(bm_autodiff_dual PRIVATE ${TOOLS_INSTALL_DIR}/include)
  target_link_directories(bm_autodiff_dual PRIVATE ${TOOLS_INSTALL_DIR}/lib)
  target_link_libraries(bm_autodiff_dual autodiff::autodiff Eigen3::Eigen pthread)
  target_compile_definitions(bm_autodiff_dual PRIVATE ENABLE_AUTODIFF_DUAL)

  add_executable(bm_autodiff_real main.cpp)
  target_include_directories(bm_autodiff_real PRIVATE ${TOOLS_INSTALL_DIR}/include)
  target_link_directories(bm_autodiff_real PRIVATE ${TOOLS_INSTALL_DIR}/lib)
  target_link_libraries(bm_autodiff_real autodiff::autodiff Eigen3::Eigen pthread)
  target_compile_definitions(bm_autodiff_real PRIVATE ENABLE_AUTODIFF_REAL)

  add_executable(bm_autodiff_var main.cpp)
  target_include_directories(bm_autodiff_var PRIVATE ${TOOLS_INSTALL_DIR}/include)
  target_link_directories(bm_autodiff_var PRIVATE ${TOOLS_INSTALL_DIR}/lib)
  target_link_libraries(bm_autodiff_var autodiff::autodiff Eigen3::Eigen pthread)
  target_compile_definitions(bm_autodiff_var PRIVATE ENABLE_AUTODIFF_VAR)
endif()

# adept benchmark
if(EXISTS ${TOOLS_INSTALL_DIR}/AD_BENCHMARK_ADEPT)
  add_executable(bm_adept main.cpp)
  target_include_directories(bm_adept PRIVATE ${TOOLS_INSTALL_DIR}/include)
  target_link_directories(bm_adept PRIVATE ${TOOLS_INSTALL_DIR}/lib)
  target_link_libraries(bm_adept adept Eigen3::Eigen pthread)
  target_compile_definitions(bm_adept PRIVATE ENABLE_ADEPT)
endif()

# # adolc benchmark
if(EXISTS ${TOOLS_INSTALL_DIR}/AD_BENCHMARK_ADOLC)
  add_executable(bm_adolc main.cpp)
  target_include_directories(bm_adolc PRIVATE ${TOOLS_INSTALL_DIR}/include)
  target_link_directories(bm_adolc PRIVATE ${TOOLS_INSTALL_DIR}/lib)
  target_link_libraries(bm_adolc adolc Eigen3::Eigen pthread)
  target_compile_definitions(bm_adolc PRIVATE ENABLE_ADOLC)
endif()

# # ceres benchmark
find_package(Ceres QUIET PATHS ${TOOLS_INSTALL_DIR}/lib/cmake)
if(${Ceres_FOUND})
  add_executable(bm_ceres main.cpp)
  target_include_directories(bm_ceres PRIVATE ${TOOLS_INSTALL_DIR}/include)
  target_link_directories(bm_ceres PRIVATE ${TOOLS_INSTALL_DIR}/lib)
  target_link_libraries(bm_ceres Ceres::ceres Eigen3::Eigen pthread)
  target_compile_definitions(bm_ceres PRIVATE ENABLE_CERES)
endif()

if(EXISTS ${TOOLS_INSTALL_DIR}/AD_BENCHMARK_CPPAD)
  add_executable(bm_cppad main.cpp)
  target_include_directories(bm_cppad PRIVATE ${TOOLS_INSTALL_DIR}/include)
  target_link_directories(bm_cppad PRIVATE ${TOOLS_INSTALL_DIR}/lib)
  target_link_libraries(bm_cppad cppad_lib Eigen3::Eigen pthread)
  target_compile_definitions(bm_cppad PRIVATE ENABLE_CPPAD)
endif()

if(EXISTS ${TOOLS_INSTALL_DIR}/AD_BENCHMARK_CPPADCG)
  add_executable(bm_cppadcg main.cpp)
  target_include_directories(bm_cppadcg PRIVATE ${TOOLS_INSTALL_DIR}/include)
  target_link_directories(bm_cppadcg PRIVATE ${TOOLS_INSTALL_DIR}/lib)
  target_link_libraries(bm_cppadcg cppad_lib dl Eigen3::Eigen pthread)
  target_compile_definitions(bm_cppadcg PRIVATE ENABLE_CPPADCG CPPAD_FOR_TMB)  # flag required for SE3 ODE test
endif()

find_package(Sacado QUIET PATHS ${TOOLS_INSTALL_DIR}/lib/cmake)
if(${Sacado_FOUND})
  add_executable(bm_sacado main.cpp)
  target_include_directories(bm_sacado PRIVATE ${TOOLS_INSTALL_DIR}/include)
  target_link_directories(bm_sacado PRIVATE ${TOOLS_INSTALL_DIR}/lib)
  target_link_libraries(bm_sacado sacado Eigen3::Eigen pthread)
  target_compile_definitions(bm_sacado PRIVATE ENABLE_SACADO)
endif()

