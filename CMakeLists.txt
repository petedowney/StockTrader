cmake_minimum_required(VERSION 3.0)
project(NNProject)


find_package(PkgConfig)

pkg_check_modules(TensorFlow REQUIRED tensorflow)
pkg_check_modules(Eigen3 REQUIRED eigen3)

link_directories(${TensorFlow_LIBRARY_DIRS})
link_directories(${Eigen3_LIBRARY_DIRS})
include_directories(${TensorFlow_INCLUDE_DIRS})

# Setup Tensorflow C++
set(TENSORFLOW_INCLUDES
        /usr/local/include/tf/
        /usr/local/include/tf/bazel-genfiles
        /usr/local/include/tf/tensorflow/
        /usr/local/include/tf/tensorflow/third-party
        )

set(TENSORFLOW_LIBS
        /usr/local/lib/libtensorflow_cc.so
        /usr/local/lib/libtensorflow_framework.so)

include_directories(
        ${TENSORFLOW_INCLUDES}
        ${PROJECT_SOURCE_DIR}/third_party/eigen3
)

# Simple Net C++


set(CMAKE_CXX_STANDARD 20)

add_executable(NNProject main.cpp model/neuralNet.cpp model/neuralNet.h)
