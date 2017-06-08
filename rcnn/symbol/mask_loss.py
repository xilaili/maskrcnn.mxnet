# -------------------------
# Mask loss
# -------------------------
import mxnet as mx
import numpy as np


class MaskLossOperator(mx.operator.CustomOp):
    def __init__(self, num_classes):
        super(MaskLossOperator, self).__init__()
        self._num_classes = num_classes

    def forward(self, is_train, req, in_data, out_data, aux):
        mask_reg_targets = in_data[0].asnumpy()
        mask_pred = in_data[1].asnumpy()



    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):


@mx.operator.register('mask_loss')
class MaskLossProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes):
        super(MaskLossProp, self).__init__(need_top_grad=False)
        self._num_classes = num_classes

    def list_arguments(self):
        return ['mask_reg_targets', 'mask_pred']

    def list_outputs(self):
        return ['mask_loss']

    def infer_shape(self, in_shape):

    def create_operator(self, ctx, shapes, dtypes):
        return MaskLossOperator(self._num_classes)


    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []