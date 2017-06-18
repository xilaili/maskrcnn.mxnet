import numpy as np
import mxnet as mx
import random

data = mx.symbol.Variable(name='data')
rois = mx.symbol.Variable(name='rois')
#test = mx.symbol.ROIPooling(data=data, rois=rois, pooled_size=(2, 2), spatial_scale=1)
test = mx.symbol.ROIAlign(data=data, rois=rois, pooled_size=(3, 3), spatial_scale=1)
'''
x = [[[[  0.,   1.,   2.,   3.,   4.,   5.],
       [  6.,   7.,   8.,   9.,  10.,  11.],
       [ 12.,  13.,  14.,  15.,  16.,  17.],
       [ 18.,  19.,  20.,  21.,  22.,  23.],
       [ 24.,  25.,  26.,  27.,  28.,  29.],
       [ 30.,  31.,  32.,  33.,  34.,  35.],
       [ 36.,  37.,  38.,  39.,  40.,  41.],
       [ 42.,  43.,  44.,  45.,  46.,  47.]]]]
'''
h, w = 10, 10
x = [[[[i*w+j for j in range(w)] for i in range(h)]]]
for x_ in x[0][0]:
    print x_

y = [[0,0,0,1,1]]


x = mx.nd.array(x, dtype='float32')
y = mx.nd.array(y, dtype='float32')

ex = test.bind(ctx=mx.cpu(), args={'data':x, 'rois':y})
y = ex.forward()
print y
print y[0].asnumpy()

