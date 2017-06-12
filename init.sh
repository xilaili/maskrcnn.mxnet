#!/bin/bash

cd rcnn/cython
python setup.py build_ext --inplace
cd ../dataset/pycocotools
python setup.py build_ext --inplace
cd ../mask
python setup_linux.py build_ext --inplace
cd ../..
