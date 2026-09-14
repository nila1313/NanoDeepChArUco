
cmake_minimum_required(VERSION 3.20)
project(nanodeep_directory_detector LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs objdetect aruco)
add_executable(detect_batch main.cpp)
target_include_directories(detect_batch PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(detect_batch PRIVATE ${OpenCV_LIBS})
