cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(image_catching)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "config: ${OpenCV_DIR}")

add_executable(image_catching main.cpp)

target_link_libraries(image_catching PUBLIC ${OpenCV_LIBS})
