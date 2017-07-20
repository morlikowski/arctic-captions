#! /bin/bash

# TODO first try to copy from existing installation 

mkdir .opencv_download
cd .opencv_download

wget https://github.com/opencv/opencv/archive/2.4.13.2.zip

unzip 2.4.13.2.zip

cd opencv-2.4.13.2/
mkdir release
cd release

cmake -D MAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/local/ -D PYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python/ -D PYTHON_PACKAGES_PATH=$VIRTUAL_ENV/lib/python2.7/site-packages/ ..

make -j8
make install
