/*!
 * Copyright (c) 2017 by Contributors
 * \file roi_align.cu
 * \brief roi align operator
 * \author Xilai
*/
#include "./roi_align-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>

namespace mshadow {
namespace cuda {

template<typename Dtype>
__global__ void ROIAlignForwardKernel(const int count, const Dtype* bottom_data,
                                     const float spatial_scale, const int channels,
                                     const int height, const int width,
                                     const int pooled_height, const int pooled_width,
                                     const Dtype* bottom_rois, Dtype* top_data,
                                     Dtype* argmax_data) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];

    if (roi_batch_ind < 0) {
      top_data[index] = 0;
      argmax_data[index] = 0;
      continue;
    }

    float roi_start_w = bottom_rois[1] * spatial_scale;
    float roi_start_h = bottom_rois[2] * spatial_scale;
    float roi_end_w = bottom_rois[3] * spatial_scale;
    float roi_end_h = bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    float roi_width = fmaxf(roi_end_w - roi_start_w, 0);
    //float roi_width = fmaxf(roi_end_w - roi_start_w + 1, 0);
    float roi_height = fmaxf(roi_end_h - roi_start_h, 0);
    //float roi_height = fmaxf(roi_end_h - roi_start_h + 1, 0);
    float bin_size_h = roi_height / (pooled_height - 1);
    float bin_size_w = roi_width / (pooled_width - 1);

    float h_ = float(ph) * bin_size_h + roi_start_h;
    float w_ = float(pw) * bin_size_w + roi_start_w;
    int hstart = fminf(floor(h_), height-2);
    int wstart = fminf(floor(w_), width-2);

    if (h_<0 || h_>=height || w_<0 || w_>=width) {
      top_data[index] = 0;
    } else {
      bottom_data += (roi_batch_ind * channels + c) * height * width;
      float  h_ratio = h_ - (float)(hstart);
      float  w_ratio = w_ - (float)(wstart);
      int upleft = hstart * width + wstart;
      int upright = upleft + 1;
      int downleft = upleft + width;
      int downright = downleft + 1;

      top_data[index] = bottom_data[upleft]*(1.-h_ratio)*(1.-w_ratio)
                           + bottom_data[upright]*(1.-h_ratio)*w_ratio
                           + bottom_data[downleft]*h_ratio*(1.-w_ratio)
                           + bottom_data[downright]*h_ratio*w_ratio;
    }
  }
}

template<typename Dtype>
inline void ROIAlignForward(const Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 4, Dtype> &max_idx,
                           const float spatial_scale) {
  const Dtype *bottom_data = data.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *top_data = out.dptr_;
  Dtype *argmax_data = max_idx.dptr_;
  const int count = out.shape_.Size();
  const int channels = data.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int pooled_height = out.size(2);
  const int pooled_width = out.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "ROIAlign Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  ROIAlignForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, bottom_data, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_rois, top_data, argmax_data);
}

template<typename Dtype>
__global__ void ROIAlignBackwardAccKernel(const int count, const Dtype* top_diff,
                                         const Dtype* argmax_data, const int num_rois,
                                         const float spatial_scale, const int channels,
                                         const int height, const int width,
                                         const int pooled_height, const int pooled_width,
                                         Dtype* bottom_diff, const Dtype* bottom_rois) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {

    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];

    if (roi_batch_ind < 0) {
      bottom_diff[index] = 0;
      continue;
    }

    float roi_start_w = bottom_rois[1] * spatial_scale;
    float roi_start_h = bottom_rois[2] * spatial_scale;
    float roi_end_w = bottom_rois[3] * spatial_scale;
    float roi_end_h = bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    float roi_width = fmaxf(roi_end_w - roi_start_w, 0);
    //float roi_width = fmaxf(roi_end_w - roi_start_w + 1, 0);
    float roi_height = fmaxf(roi_end_h - roi_start_h, 0);
    //float roi_height = fmaxf(roi_end_h - roi_start_h + 1, 0);
    float bin_size_h = roi_height / (pooled_height - 1);
    float bin_size_w = roi_width / (pooled_width - 1);

    float h_ = float(ph) * bin_size_h + roi_start_h;
    float w_ = float(pw) * bin_size_w + roi_start_w;
    int hstart = fminf(floor(h_), height-2);
    int wstart = fminf(floor(w_), width-2);

    if (h_>=0 && h_<height && w_>=0 && w_<width) {
      bottom_diff += (roi_batch_ind * channels + c) * height * width;
      float  h_ratio = h_ - (float)(hstart);
      float  w_ratio = w_ - (float)(wstart);
      int upleft = hstart * width + wstart;
      int upright = upleft + 1;
      int downleft = upleft + width;
      int downright = downleft + 1;

      atomicAdd(bottom_diff + upleft, top_diff[index]*(1.-h_ratio)*(1.-w_ratio));
      atomicAdd(bottom_diff + upright, top_diff[index]*(1.-h_ratio)*w_ratio);
      atomicAdd(bottom_diff + downleft, top_diff[index]*h_ratio*(1.-w_ratio));
      atomicAdd(bottom_diff + downright, top_diff[index]*h_ratio*w_ratio);
    }

  }
}

template<typename Dtype>
inline void ROIAlignBackwardAcc(const Tensor<gpu, 4, Dtype> &in_grad,
                               const Tensor<gpu, 4, Dtype> &out_grad,
                               const Tensor<gpu, 2, Dtype> &bbox,
                               const Tensor<gpu, 4, Dtype> &max_idx,
                               const float spatial_scale) {
  const Dtype *top_diff = out_grad.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *bottom_diff = in_grad.dptr_;
  Dtype *argmax_data = max_idx.dptr_;
  const int count = in_grad.shape_.Size();
  const int num_rois = bbox.size(0);
  const int channels = in_grad.size(1);
  const int height = in_grad.size(2);
  const int width = in_grad.size(3);
  const int pooled_height = out_grad.size(2);
  const int pooled_width = out_grad.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "ROIAlign Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
  ROIAlignBackwardAccKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, top_diff, argmax_data, num_rois, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_diff, bottom_rois);
}

}  // namespace cuda

template<typename Dtype>
inline void ROIAlignForward(const Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 4, Dtype> &max_idx,
                           const float spatial_scale) {
  cuda::ROIAlignForward(out, data, bbox, max_idx, spatial_scale);
}

template<typename Dtype>
inline void ROIAlignBackwardAcc(const Tensor<gpu, 4, Dtype> &in_grad,
                               const Tensor<gpu, 4, Dtype> &out_grad,
                               const Tensor<gpu, 2, Dtype> &bbox,
                               const Tensor<gpu, 4, Dtype> &max_idx,
                               const float spatial_scale) {
  cuda::ROIAlignBackwardAcc(in_grad, out_grad, bbox, max_idx, spatial_scale);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(ROIAlignParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ROIAlignOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
