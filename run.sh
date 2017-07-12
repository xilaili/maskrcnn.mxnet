#!/bin/bash
python maskrcnn_train_end2end.py --gpus 1 --prefix model/e2e_pool_fullcoco_0710
#python maskrcnn_demo.py --gpu 1 --prefix model/e2e_pool_fullcoco_0704 --epoch 20
