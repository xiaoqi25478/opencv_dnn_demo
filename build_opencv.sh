set -e
set -x

PRJ_PATH=$PWD

cd ../

wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip

wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv_contrib.zip

cd opencv-4.x

mkdir -p build && cd build

cmake -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.x/modules .. \
    -DWITH_CUDA=1 \
    -DCUDA_ARCH_BIN=8.6 \
    -DENABLE_FAST_MATH=1 \
    -DCUDA_FAST_MATH=1 \
    -DWITH_CUBLAS=1 \
    -DOPENCV_GENERATE_PKGCONFIG=1 \
    ..

make -j100
make install

cd $PRJ_PATH

