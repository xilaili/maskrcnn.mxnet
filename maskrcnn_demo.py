# require opencv 3.2
import argparse
import os
import sys
from rcnn.logger import logger
import cv2
from rcnn.config import config
from rcnn.symbol import get_resnet_test
from rcnn.io.image import resize, transform
import numpy as np
# get config
#os.environ['PYTHONUNBUFFERED'] = '1'
#os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
#os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

import mxnet as mx
from rcnn.core.tester import im_detect, Predictor
from rcnn.utils.load_model import load_param
#from rcnn.utils.show_masks import show_masks
from rcnn.utils.tictoc import tic, toc
from rcnn.utils.show_masks import show_masks
from rcnn.processing.nms import py_nms_wrapper
from rcnn.mask.mask_transform import gpu_mask_voting, cpu_mask_voting
from rcnn.processing.bbox_transform import clip_boxes


def main():
    # get symbol
    ctx = mx.gpu(0)
    sym = get_resnet_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)

    # set up class names
    num_classes = 81
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    # load demo data
    image_names = ['COCO_test2015_000000000275.jpg', 'COCO_test2015_000000001412.jpg', 'COCO_test2015_000000073428.jpg',
                    'COCO_test2015_000000393281.jpg']
    data = []
    for im_name in image_names:
        assert os.path.exists('../data/demo/' + im_name), ('%s does not exist'.format('../data/demo/' + im_name))
        im = cv2.imread('../data/demo/' + im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.IMAGE_STRIDE)
        im_tensor = transform(im, config.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})

    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    arg_params, aux_params = load_param('model/e2e', 1, process=True)
    print data_names, label_names, ctx, max_data_shape, provide_data, provide_label
    predictor = Predictor(sym, data_names, label_names,
                          context=[ctx], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # warm up
    for i in xrange(2):
        data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
        _, _, _, _ = im_detect(predictor, data_batch, data_names, scales)

    # test
    for idx, im_name in enumerate(image_names):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        tic()
        scores, boxes, masks, data_dict = im_detect(predictor, data_batch, data_names, scales)
        im_shapes = [data_batch.data[i][0].shape[2:4] for i in xrange(len(data_batch.data))]

        if not config.TEST.USE_MASK_MERGE:
            all_boxes = [[] for _ in xrange(num_classes)]
            all_masks = [[] for _ in xrange(num_classes)]
            nms = py_nms_wrapper(config.TEST.NMS)
            for j in range(1, num_classes):
                indexes = np.where(scores[0][:, j] > 0.7)[0]
                cls_scores = scores[0][indexes, j, np.newaxis]
                cls_masks = masks[0][indexes, 1, :, :]
                try:
                    if config.CLASS_AGNOSTIC:
                        cls_boxes = boxes[0][indexes, :]
                    else:
                        raise Exception()
                except:
                    cls_boxes = boxes[0][indexes, j * 4:(j + 1) * 4]

                cls_dets = np.hstack((cls_boxes, cls_scores))
                print cls_dets
                keep = nms(cls_dets)
                all_boxes[j] = cls_dets[keep, :]
                all_masks[j] = cls_masks[keep, :]
            dets = [all_boxes[j] for j in range(1, num_classes)]
            masks = [all_masks[j] for j in range(1, num_classes)]
        else:
            masks = masks[0][:, 1:, :, :]
            im_height = np.round(im_shapes[0][0] / scales[0]).astype('int')
            im_width = np.round(im_shapes[0][1] / scales[0]).astype('int')
            print (im_height, im_width)
            boxes = clip_boxes(boxes[0], (im_height, im_width))
            result_masks, result_dets = gpu_mask_voting(masks, boxes, scores[0], num_classes,
                                                        100, im_width, im_height,
                                                        config.TEST.NMS, config.TEST.MASK_MERGE_THRESH,
                                                        config.BINARY_THRESH, 0)

            dets = [result_dets[j] for j in range(1, num_classes)]
            masks = [result_masks[j][:, 0, :, :] for j in range(1, num_classes)]
        print 'testing {} {:.4f}s'.format(im_name, toc())
        # visualize
        for i in xrange(len(dets)):
            keep = np.where(dets[i][:,-1]>0.7)
            dets[i] = dets[i][keep]
            masks[i] = masks[i][keep]
        im = cv2.imread('../data/demo/' + im_name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        show_masks(im, dets, masks, classes)

    print 'done'

if __name__ == '__main__':
    main()
