import cPickle
import cv2
import os
import json
import numpy as np
import hickle as hkl
import multiprocessing as mp

from ..logger import logger
from imdb import IMDB

# coco api
from ..pycocotools.coco import COCO
from ..pycocotools.cocoeval import COCOeval
from ..pycocotools import mask as COCOmask

from ..utils.tictoc import tic, toc
from ..utils.mask_coco2voc import mask_coco2voc
from ..utils.mask_voc2coco import mask_voc2coco

def generate_cache_seg_inst_kernel(annWithObjs):
    """
    generate cache_seg_inst
    :param annWithObjs: tuple of anns and objs
    """
    ann = annWithObjs[0]      # the dictionary
    objs = annWithObjs[1]     # objs
    gt_mask_file = ann['cache_seg_inst']
    if not gt_mask_file:
        return
    gt_mask_flip_file = os.path.join(os.path.splitext(gt_mask_file)[0] + '_flip.hkl')
    if os.path.exists(gt_mask_file) and os.path.exists(gt_mask_flip_file):
        return
    gt_mask_encode = [x['segmentation'] for x in objs]
    gt_mask = mask_coco2voc(gt_mask_encode, ann['height'], ann['width'])
    if not os.path.exists(gt_mask_file):
        hkl.dump(gt_mask.astype('bool'), gt_mask_file, mode='w', compression='gzip')
    # cache flip gt_masks
    if not os.path.exists(gt_mask_flip_file):
        hkl.dump(gt_mask[:, :, ::-1].astype('bool'), gt_mask_flip_file, mode='w', compression='gzip')


class coco(IMDB):
    def __init__(self, image_set, root_path, data_path, result_path=None, mask_size=-1, binary_thresh=None):
        """
        fill basic information to initialize imdb
        :param image_set: train2014, val2014, test2015
        :param root_path: 'data', will write 'rpn_data', 'cache'
        :param data_path: 'data/coco'
        """
        super(coco, self).__init__('COCO', image_set, root_path, data_path, result_path)
        self.root_path = root_path
        self.data_path = data_path
        self.coco = COCO(self._get_ann_file())

        # deal with class names
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls], self._class_to_ind[cls])
                                            for cls in self.classes[1:]])

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('%s num_images %d' % (self.name, self.num_images))

        # deal with data name
        view_map = {'minival2014': 'val2014',
                    'valminusminival2014': 'val2014'}
        self.data_name = view_map[image_set] if image_set in view_map else image_set

    def _get_ann_file(self):
        """ self.data_path / annotations / instances_train2014.json """
        prefix = 'instances' if 'test' not in self.image_set else 'image_info'
        return os.path.join(self.data_path, 'annotations',
                            prefix + '_' + self.image_set + '.json')

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def image_path_from_index(self, index):
        """ example: images / train2014 / COCO_train2014_000000119993.jpg """
        filename = 'COCO_%s_%012d.jpg' % (self.data_name, index)
        image_path = os.path.join(self.data_path, self.data_name, filename)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            logger.info('%s gt roidb loaded from %s' % (self.name, cache_file))
            return roidb

        gt_roidb = [self._load_coco_annotation(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        logger.info('%s wrote gt roidb to %s' % (self.name, cache_file))

        return gt_roidb

    def gt_sdsdb(self):
        """
        :return:
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_sdsdb.pkl')
        """
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                sdsdb = cPickle.load(fid)
            print '{} gt sdsdb loaded from {}'.format(self.name, cache_file)
            return sdsdb
        """
        # for internal useage
        tic();
        gt_sdsdb_temp = [self.load_coco_sds_annotation(index) for index in self.image_set_index[:250]]
        gt_sdsdb = [x[0] for x in gt_sdsdb_temp]
        print 'prepare gt_sdsdb using', toc(), 'seconds';
        #objs = [x[1] for x in gt_sdsdb_temp]
        tic();
        generate_cache_seg_inst_kernel(gt_sdsdb_temp[0])
        pool = mp.Pool(mp.cpu_count())
        pool.map(generate_cache_seg_inst_kernel, gt_sdsdb_temp)
        pool.close()
        pool.join()
        print 'generate cache_seg_inst using', toc(), 'seconds';

        """
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_sdsdb, fid, cPickle.HIGHEST_PROTOCOL)
        """
        # for future release usage
        # need to implement load sbd data
        return gt_sdsdb


    def _load_coco_annotation(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: roidb entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=None)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        for ix, obj in enumerate(objs):
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            if obj['iscrowd']:
                overlaps[ix, :] = -1.0
            else:
                overlaps[ix, cls] = 1.0

        roi_rec = {'image': self.image_path_from_index(index),
                   'height': height,
                   'width': width,
                   'boxes': boxes,
                   'gt_classes': gt_classes,
                   'gt_overlaps': overlaps,
                   'max_classes': overlaps.argmax(axis=1),
                   'max_overlaps': overlaps.max(axis=1),
                   'flipped': False}
        return roi_rec

    def mask_path_from_index(self, index):
        """
        given image index, cache high resolution mask and return full path of masks
        :param index: index of a specific image
        :return: full path of this mask
        """
        if self.data_name == 'val':
            return []
        cache_file = os.path.join(self.cache_path, 'COCOMask', self.data_name)
        if not os.path.exists(cache_file):
            os.makedirs(cache_file)
        # instance level segmentation
        filename = 'COCO_%s_%012d' % (self.data_name, index)
        gt_mask_file = os.path.join(cache_file, filename + '.hkl')
        return gt_mask_file


    def load_coco_sds_annotation(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: roidb entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        # only load objs whose iscrowd==false
        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        for ix, obj in enumerate(objs):
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            if obj['iscrowd']:
                overlaps[ix, :] = -1.0
            else:
                overlaps[ix, cls] = 1.0

        sds_rec = {'image': self.image_path_from_index(index),
                   'height': height,
                   'width': width,
                   'boxes': boxes,
                   'gt_classes': gt_classes,
                   'gt_overlaps': overlaps,
                   'max_classes': overlaps.argmax(axis=1),
                   'max_overlaps': overlaps.max(axis=1),
                   'cache_seg_inst': self.mask_path_from_index(index),
                   'flipped': False}
        return sds_rec, objs


    def evaluate_detections(self, detections):
        """ detections_val2014_results.json """
        res_folder = os.path.join(self.result_path, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(res_folder, 'detections_%s_results.json' % self.image_set)
        self._write_coco_results(detections, res_file)
        if 'test' not in self.image_set:
            self._do_python_eval(res_file, res_folder)

    def _write_coco_results(self, detections, res_file):
        """ example results
        [{"image_id": 42,
          "category_id": 18,
          "bbox": [258.15,41.29,348.26,243.78],
          "score": 0.236}, ...]
        """
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            logger.info('collecting %s results (%d/%d)' % (cls, cls_ind, self.num_classes - 1))
            coco_cat_id = self._class_to_coco_ind[cls]
            results.extend(self._coco_results_one_category(detections[cls_ind], coco_cat_id))
        logger.info('writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.image_set_index):
            dets = boxes[im_ind].astype(np.float)
            if len(dets) == 0:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            result = [{'image_id': index,
                       'category_id': cat_id,
                       'bbox': [xs[k], ys[k], ws[k], hs[k]],
                       'score': scores[k]} for k in xrange(dets.shape[0])]
            results.extend(result)
        return results

    def _do_python_eval(self, res_file, res_folder):
        ann_type = 'bbox'
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._print_detection_metrics(coco_eval)

        eval_file = os.path.join(res_folder, 'detections_%s_results.pkl' % self.image_set)
        with open(eval_file, 'wb') as f:
            cPickle.dump(coco_eval, f, cPickle.HIGHEST_PROTOCOL)
        logger.info('eval results saved to %s' % eval_file)

    def _print_detection_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)

        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        logger.info('~~~~ Mean and per-category AP @ IoU=%.2f,%.2f] ~~~~' % (IoU_lo_thresh, IoU_hi_thresh))
        logger.info('%-15s %5.1f' % ('all', 100 * ap_default))
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            logger.info('%-15s %5.1f' % (cls, 100 * ap))

        logger.info('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()
