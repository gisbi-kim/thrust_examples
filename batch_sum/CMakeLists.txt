cmake_minimum_required(VERSION  3.22)

project(ThrustCpuGpuExample CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(THRUST_REQUIRED_SYSTEMS CPP CUDA)
set(THRUST_OPTIONAL_SYSTEMS CUDA)

find_package(Thrust REQUIRED CONFIG)

######
# 1. CPU: the device CPP means cpu
######
thrust_create_target(ThrustCPP HOST CPP DEVICE CPP)
add_executable(main_cpu main_cpu_std.cpp)
target_link_libraries(main_cpu ThrustCPP)

######
# 2. GPU: the device CUDA means gpu
######
thrust_update_system_found_flags()
enable_language(CUDA)
thrust_create_target(ThrustCUDA HOST CPP DEVICE CUDA)
add_executable(main_gpu main_gpu_thrust.cu)
target_link_libraries(main_gpu ThrustCUDA)

# ps. 
# for the case of if not being automatically detected..
# target_include_directories(main_gpu PRIVATE /usr/local/cuda-11.4/targets/x86_64-linux/include)
# target_link_directories(main_gpu PRIVATE /usr/local/cuda-11.4/targets/x86_64-linux/lib)

# ps. 
#### cmake .. results may be like 
# -- The CXX compiler identification is GNU 9.4.0
# -- The CUDA compiler identification is NVIDIA 11.4.48
# -- Detecting CXX compiler ABI info
# -- Detecting CXX compiler ABI info - done
# -- Check for working CXX compiler: /usr/bin/g++-9 - skipped
# -- Detecting CXX compile features
# -- Detecting CXX compile features - done
# -- Detecting CUDA compiler ABI info
# -- Detecting CUDA compiler ABI info - done
# -- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
# -- Detecting CUDA compile features
# -- Detecting CUDA compile features - done
# -- Found libcudacxx: /usr/local/lib/cmake/libcudacxx/libcudacxx-config.cmake (found suitable version "1.8.1.0", minimum required is "1.8.0") 
# -- Found Thrust: /usr/local/lib/cmake/thrust/thrust-config.cmake (found version "2.2.0.0") 
# -- Found CUB: /usr/local/lib/cmake/cub/cub-config.cmake (found suitable version "2.2.0.0", minimum required is "2.2.0.0") 
# -- Configuring done
# -- Generating done
# -- Build files have been written to: /examples/build
