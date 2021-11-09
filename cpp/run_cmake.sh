rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=../.. -DCMAKE_CXX_FLAGS="-O2 -Wall -fPIC -std=c++17" -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DCMAKE_BUILD_TYPE=Release
make -j4