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
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

import mxnet as mx
from rcnn.core.tester import im_detect, Predictor, vis_all_detection
from rcnn.utils.load_model import load_param
#from rcnn.utils.show_masks import show_masks
from rcnn.utils.tictoc import tic, toc
from rcnn.utils.show_masks import show_masks
from rcnn.processing.nms import py_nms_wrapper
from rcnn.mask.mask_transform import gpu_mask_voting, cpu_mask_voting
from rcnn.processing.bbox_transform import clip_boxes

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
config.NUM_CLASSES = 81
DATA_NAMES = ['data', 'im_info']
LABEL_NAMES = []
CONF_THRESH = 0.9
NMS_THRESH = 0.3

def load_data():
    # load demo data
    image_names = ['COCO_test2015_000000000275.jpg', 'COCO_test2015_000000001412.jpg', 'COCO_test2015_000000073428.jpg',
                    'COCO_test2015_000000393281.jpg']
    data = []
    im_scales = []
    for im_name in image_names:
        assert os.path.exists('../data/demo/' + im_name), ('%s does not exist'.format('../data/demo/' + im_name))
        im = cv2.imread('../data/demo/' + im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.IMAGE_STRIDE)
        im_tensor = transform(im, config.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})
        im_scales.append(im_scale)

    return data, image_names, im_scales

def get_net(data, sym, prefix, epoch, ctx):
    # get predictor
    data = [[mx.nd.array(data[i][name]) for name in DATA_NAMES] for i in xrange(len(data))]
    max_data_shape = [('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    provide_data = [[(k, v.shape) for k, v in zip(DATA_NAMES, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    arg_params, aux_params = load_param(prefix, epoch, process=True)
    print DATA_NAMES, LABEL_NAMES, ctx, max_data_shape, provide_data, provide_label
    predictor = Predictor(sym, DATA_NAMES, LABEL_NAMES,
                          context=[ctx], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    return predictor

def demo_net(predictor, data, image_names, im_scales):
    data = [[mx.nd.array(data[i][name]) for name in DATA_NAMES] for i in xrange(len(data))]
    # warm up
    for i in xrange(2):
        data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(DATA_NAMES, data[0])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
        _, _, _, _, _= im_detect(predictor, data_batch, DATA_NAMES, scales)

    # test
    for idx, im_name in enumerate(image_names):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(DATA_NAMES, data[idx])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        tic()
        scores, boxes, boxes2, masks, data_dict = im_detect(predictor, data_batch, DATA_NAMES, scales)
        im_shapes = [data_batch.data[i][0].shape[2:4] for i in xrange(len(data_batch.data))]

        print boxes[0].shape
        # mask output
        if not config.TEST.USE_MASK_MERGE:
            all_boxes = [[] for _ in xrange(config.NUM_CLASSES)]
            all_masks = [[] for _ in xrange(config.NUM_CLASSES)]
            nms = py_nms_wrapper(config.TEST.NMS)
            for j in range(1, config.NUM_CLASSES):
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
            dets = [all_boxes[j] for j in range(1, config.NUM_CLASSES)]
            masks = [all_masks[j] for j in range(1, config.NUM_CLASSES)]
        else:
            masks = masks[0][:, 1:, :, :]
            im_height = np.round(im_shapes[0][0] / scales[0]).astype('int')
            im_width = np.round(im_shapes[0][1] / scales[0]).astype('int')
            print (im_height, im_width)
            boxes_ = clip_boxes(boxes[0], (im_height, im_width))
            result_masks, result_dets = gpu_mask_voting(masks, boxes_, scores[0], config.NUM_CLASSES,
                                                        100, im_width, im_height,
                                                        config.TEST.NMS, config.TEST.MASK_MERGE_THRESH,
                                                        config.BINARY_THRESH, 0)

            dets = [result_dets[j] for j in range(1, config.NUM_CLASSES)]
            masks = [result_masks[j][:, 0, :, :] for j in range(1, config.NUM_CLASSES)]
        print 'testing {} {:.4f}s'.format(im_name, toc())
        # visualize
        for i in xrange(len(dets)):
            keep = np.where(dets[i][:,-1]>0.7)
            dets[i] = dets[i][keep]
            masks[i] = masks[i][keep]
        im = cv2.imread('../data/demo/' + im_name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        show_masks(im, dets, masks, CLASSES)

        # bounding box output
        all_boxes = [[] for _ in CLASSES]
        nms = py_nms_wrapper(NMS_THRESH)
        for cls in CLASSES:
            cls_ind = CLASSES.index(cls)
            cls_boxes = boxes2[0][:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[0][:, cls_ind, np.newaxis]
            keep = np.where(cls_scores >= CONF_THRESH)[0]
            dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
            keep = nms(dets)
            all_boxes[cls_ind] = dets[keep, :]

        boxes_this_image = [[]] + [all_boxes[j] for j in range(len(CLASSES))]
        vis_all_detection(data_dict[0]['data'].asnumpy(), boxes_this_image, CLASSES, im_scales[idx])

    print 'done'

def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate a Mask R-CNN network')
    parser.add_argument('--prefix', help='saved model prefix', default='model/e2e', type=str)
    parser.add_argument('--epoch', help='epoch of pretrained model', default=0, type=int)
    parser.add_argument('--gpu', help='GPU device to use', default=0, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    ctx = mx.gpu(args.gpu)
    sym = get_resnet_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    data, image_names, im_scales = load_data()
    predictor = get_net(data, sym, args.prefix, args.epoch, ctx)
    demo_net(predictor, data, image_names, im_scales)

if __name__ == '__main__':
    main()
