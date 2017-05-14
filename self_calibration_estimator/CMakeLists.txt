cmake_minimum_required(VERSION 2.8.3)
project(self_calibration_estimator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -DEIGEN_DONT_PARALLELIZE")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    )

find_package(OpenCV REQUIRED)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIR})
#include_directories(/home/dvorak/code/Intel/mkl/include)
#link_directories(/home/dvorak/code/Intel/mkl/lib/intel64_lin)

catkin_package()

add_executable(self_calibration_estimator
    src/self_calibration_estimator_node.cpp
    src/parameters.cpp
    src/visualization.cpp
    src/self_calibration_estimator.cpp
    src/motion_estimator.cpp
    src/marginalization.cpp
    src/feature_manager.cpp
    src/pose_local_parameterization.cpp
    src/projection_factor.cpp
    src/marginalization_factor.cpp
    src/utility.cpp
    )


target_link_libraries(self_calibration_estimator ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
#lapack mkl_intel_lp64 mkl_core mkl_sequential pthread m)

catkin_add_gtest(integration
    src/integration.cpp
    src/parameters.cpp)

target_link_libraries(integration ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
