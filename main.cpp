#include <string>

#include <cuda_runtime_api.h>
#include <cvcuda/OpResize.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "common/TestUtils.h"

void CopyAsync(cudaStream_t stream, const cv::Mat &src, nvcv::Image &dst) {
  auto data = dst.exportData<nvcv::ImageDataStridedCuda>();
  CV_Assert(data.hasValue());
  CV_Assert(src.cols == data->plane(0).width);
  CV_Assert(src.rows == data->plane(0).height);

  CHECK_CUDA_ERROR(cudaMemcpy2DAsync(
      data->plane(0).basePtr, data->plane(0).rowStride, src.data, src.step,
      src.cols * 3, src.rows, cudaMemcpyHostToDevice, stream));
}

void CopyAsync(cudaStream_t stream, const nvcv::Image &src, cv::Mat &dst) {
  auto srcData = src.exportData<nvcv::ImageDataStridedCuda>();
  CV_Assert(srcData.hasValue());

  int height = srcData->plane(0).height;
  int width = srcData->plane(0).width;

  dst.create(cv::Size2i(width, height), CV_8UC3);
  CHECK_CUDA_ERROR(cudaMemcpy2DAsync(dst.data, dst.step,
                                     srcData->plane(0).basePtr,
                                     srcData->plane(0).rowStride, width * 3,
                                     height, cudaMemcpyDeviceToHost, stream));
}

int main(int argc, char *argv[]) {
  std::string inputPath = "./cat.jpg";
  std::string resultPath = "./cat_resized.jpg";
  int resizeWidth = 320;
  int resizeHeight = 240;

  cv::Mat inputImage = cv::imread(inputPath);

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  nvcv::ImageBatchVarShape inputBatch(1);
  {
    nvcv::Image image(nvcv::Size2D{inputImage.cols, inputImage.rows},
                      nvcv::FMT_BGR8);
    CopyAsync(stream, inputImage, image);
    inputBatch.pushBack(image);
  }

  nvcv::ImageBatchVarShape resizedBatch(1);
  {
    nvcv::Image resizedImage(nvcv::Size2D{resizeWidth, resizeHeight},
                             nvcv::FMT_BGR8);
    resizedBatch.pushBack(resizedImage);
  }

  cvcuda::Resize resizeOp;
  resizeOp(stream, inputBatch, resizedBatch, NVCV_INTERP_LINEAR);

  cv::Mat result;
  CopyAsync(stream, resizedBatch[0], result);

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

  cv::imwrite(resultPath, result);
}