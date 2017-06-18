#!/bin/bash
mkdir -p model

cd rcnn/cython
python setup.py build_ext --inplace
cd ../pycocotools
python setup.py build_ext --inplace
cd ../mask
python setup_linux.py build_ext --inplace
cd ../..
