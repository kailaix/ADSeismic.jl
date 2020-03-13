cmake_minimum_required(VERSION 3.5)
project(TFApp)

##  For PyTorch
# execute_process(COMMAND julia -e "abspath(joinpath(Base.find_package(\"ADCME\"), \"../../deps/Libraries/libtorch\"))|>print" OUTPUT_VARIABLE PREFIX)
# execute_process(COMMAND julia -e "abspath(joinpath(Base.find_package(\"ADCME\"), \"../../examples/custom_op/headers\"))|>print" OUTPUT_VARIABLE HEADERS)
# execute_process(COMMAND julia -e "abspath(joinpath(Base.find_package(\"ADCME\"), \"../../deps/Libraries/libtorch/include\"))|>print" OUTPUT_VARIABLE TORCH_INC)
# set(CMAKE_PREFIX_PATH ${PREFIX})
# find_package(Torch REQUIRED)

set (CMAKE_CXX_STANDARD 11)

SET(CMAKE_C_COMPILER ${CONDAC})
SET(CMAKE_CXX_COMPILER ${CONDACPP})
execute_process(COMMAND julia -e "using PyCall; print(PyCall.python)" OUTPUT_VARIABLE PYTHON)
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_compile_flags()[0][2:])" OUTPUT_VARIABLE TF_INC)
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_link_flags()[0][2:])" OUTPUT_VARIABLE TF_LIB)
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_compile_flags()[1][-1])" OUTPUT_VARIABLE TF_ABI)
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_link_flags()[1][3:])" OUTPUT_VARIABLE TF_LIB_FILE)

message("Python path=${PYTHON}")
message("TF_INC=${TF_INC}")
message("TF_LIB=${TF_LIB}")


# https://github.com/tensorflow/tensorflow/issues/1569
# if GCC > 5
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TF_ABI} ${CMAKE_CXX_FLAGS}")
endif()

execute_process(COMMAND julia -e "abspath(joinpath(Base.find_package(\"ADCME\"), \"../../deps/Libraries\"))|>print" OUTPUT_VARIABLE EIGEN_INC)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3 -DNDEBUG")

include_directories(${TF_INC} ${HEADERS} ${EIGEN_INC} ${TORCH_INC})
link_directories(${TF_LIB})

find_package(CUDA QUIET)
SET(CUDA_PROPAGATE_HOST_FLAGS ON)


# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-shared ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fPIC ${CMAKE_CXX_FLAGS}")
SET(CUDA_PROPAGATE_HOST_FLAGS ON)


find_program(_nvidia_smi "nvidia-smi")
if (_nvidia_smi)
  cuda_add_library(GetReceive SHARED GetReceive.cu GetReceive.cpp GetReceive.cc)
else()
  add_library(GetReceive SHARED GetReceive.cpp GetReceive.cc)
  add_definitions(-DNOGPU)
endif()

set_property(TARGET GetReceive PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(GetReceive ${TF_LIB_FILE} "${TORCH_LIBRARIES}")
