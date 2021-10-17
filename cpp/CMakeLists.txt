project(dou)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_BUILD_TYPE "Debug")

find_package(Torch REQUIRED)
# find_package(PythonLibs REQUIRED)
if(MSVC)
  # THPVariable_Wrap error
  # find_library(TORCH_PYTHON_LIBRARY torch_python ${TORCH_INSTALL_PREFIX}/lib REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
endif()
# include_directories(${PYTHON_INCLUDE_DIRS})

message("")
message("CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
message("PYTHON_INCLUDE_DIRS=${PYTHON_INCLUDE_DIRS}")
message("PYTHON_LIBRARIES=${PYTHON_LIBRARIES}")
message("TORCH_LIBRARIES=${TORCH_LIBRARIES}")
message("TORCH_INSTALL_PREFIX=${TORCH_INSTALL_PREFIX}")
message("TORCH_INCLUDE_DIRS=${TORCH_INCLUDE_DIRS}")
message("TORCH_PYTHON_LIBRARY=${TORCH_PYTHON_LIBRARY}")
message("CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
message("")

add_library(game game.cpp)
target_include_directories(game PUBLIC "${PYTHON_INCLUDE_DIRS}")
target_link_libraries(game PUBLIC "${TORCH_LIBRARIES}")

# add_subdirectory("${pybind11_DIR}" pybind11 EXCLUDE_FROM_ALL)

add_executable(test test.cpp)
target_link_libraries(test PRIVATE game "${PYTHON_LIBRARIES}")

add_executable(thread_test thread_test.cpp)
target_link_libraries(thread_test PRIVATE "${TORCH_LIBRARIES}")

add_executable(dou_train train.cpp)
target_link_libraries(dou_train PRIVATE game "${PYTHON_LIBRARIES}")

add_executable(evaluate_two_agent evaluate_two_agent.cpp)
target_link_libraries(evaluate_two_agent PRIVATE game "${PYTHON_LIBRARIES}")
