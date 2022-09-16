cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(net_training)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "config: ${OpenCV_DIR}")

add_executable(net_training main.cpp)

target_link_libraries(net_training PUBLIC ${OpenCV_LIBS})