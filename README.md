## 1. Mask RCNN in MXNet:
  * Training:
     ```
     python maskrcnn_train_end2end.py --gpus 0 --prefix model/e2e  --end_epoch 10
     ```
  * Testing:
     ```
     python maskrcnn_test.py --gpu 0 --prefix model/e2e --epoch 10 --vis
     ```
  * Demo:
     ```
     python maskrcnn_demo.py --gpu 0 --prefix model/e2e --epoch 10 
     ```
