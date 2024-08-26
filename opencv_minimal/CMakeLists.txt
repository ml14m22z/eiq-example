cmake_minimum_required(VERSION 3.16)
project(minimal C CXX)


find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
add_executable(minimal
  minimal.cc
)
target_link_libraries(minimal
    ${OpenCV_LIBS}
)

add_executable(rtsp_demo
  rtps.cc
)

target_link_libraries(rtsp_demo
    ${OpenCV_LIBS}
)

