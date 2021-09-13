cmake_minimum_required(VERSION 3.19)
project(NNProject)

set(CMAKE_CXX_STANDARD 14)

include_directories("${PROJECT_SOURCE_DIR}/inc/")

#set(PYTHON_FILES temp.py)

#include_directories(${PYTHON_INCLUDE_DIRS})
#target_link_libraries(${PYTHON_LIBRARIES})


#link_libraries(/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.8/lib)
#LINK_DIRECTORIES(${PYTHON_LIBRARY})

add_executable(NNProject main.cpp)