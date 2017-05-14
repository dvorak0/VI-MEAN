cmake_minimum_required(VERSION 2.8.3)
find_package(catkin REQUIRED)
catkin_package(INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME})

find_package(cmake_modules REQUIRED)
find_package(Eigen REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++0x")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

include_directories(include)

add_library(${PROJECT_NAME}
    src/Chunk.cpp
    src/ChunkManager.cpp
    src/DistVoxel.cpp
    src/ColorVoxel.cpp
    src/geometry/AABB.cpp
    src/geometry/Plane.cpp
    src/geometry/Frustum.cpp
    src/camera/Intrinsics.cpp
    src/camera/PinholeCamera.cpp
    src/pointcloud/PointCloud.cpp
    src/ProjectionIntegrator.cpp
    src/Chisel.cpp
    src/mesh/Mesh.cpp
    src/marching_cubes/MarchingCubes.cpp
    src/io/PLY.cpp
    src/geometry/Raycast.cpp)

target_link_libraries(${PROJECT_NAME} ${CATKIN_LIBRARIES})

install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    )
