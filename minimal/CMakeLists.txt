#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Builds the minimal Tensorflow Lite example.

cmake_minimum_required(VERSION 3.16)
project(minimal C CXX)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../"
    ABSOLUTE
  )
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

set(CMAKE_CXX_STANDARD 17)

# -------------------------------------

add_executable(minimal
  minimal.cc
)
target_link_libraries(minimal
  tensorflow-lite
)

# -------------------------------------

find_package(OpenCV REQUIRED)

add_executable(dms
  dms.cc
  InputFiles.cc
  grace_hopper.cc
)

target_include_directories(dms
  PUBLIC ./
)

target_link_libraries(dms
  tensorflow-lite
  ${OpenCV_LIBS}
)

# -------------------------------------

add_executable(dms_shm
  dms_shm.cc
)

target_include_directories(dms_shm
  PUBLIC ./
)

target_link_libraries(dms_shm
  tensorflow-lite
  ${OpenCV_LIBS}
)

# -------------------------------------

add_executable(imshow
  imshow.cc
  InputFiles.cc
  grace_hopper.cc
)

target_include_directories(imshow
  PUBLIC ./
)

target_link_libraries(imshow
  ${OpenCV_LIBS}
)
