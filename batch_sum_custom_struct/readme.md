## An example result 
- See the `result.txt` file. 

## How to use 
1. We assume you already have a linux (e.g., ubuntu 20.04) image where the thrust lib is installed via the steps written in https://github.com/NVIDIA/thrust#developing-thrust (e.g., git clone yourself, cmake .., and make install). The image name is, for example, `thrust` (see docker_run.sh).
2. `$ ./docker_run.sh $(pwd)`
3. mkdir build, cd build, cmake .., make -j, 
4. run the executables e.g., `./main_cpu` and `./main_gpu`
