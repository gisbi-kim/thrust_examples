cmake_minimum_required(VERSION  3.22)

project(ThrustCpuGpuExample CXX CUDA)

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

# for the case of if not being automatically detected..
# target_include_directories(main_gpu PRIVATE /usr/local/cuda-11.4/targets/x86_64-linux/include)
# target_link_directories(main_gpu PRIVATE /usr/local/cuda-11.4/targets/x86_64-linux/lib)
