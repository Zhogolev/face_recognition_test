cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(demo)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "config: ${OpenCV_DIR}")

add_executable(demo main.cpp)

target_link_libraries(demo PUBLIC ${OpenCV_LIBS})