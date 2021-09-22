cmake_minimum_required(VERSION 3.19)
project(NNProject)

set(CMAKE_CXX_STANDARD 20)

add_executable(NNProject main.cpp model/neuralNet.cpp model/neuralNet.h)