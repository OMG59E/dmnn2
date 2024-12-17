/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 09:21:39
 * @FilePath: /dmnn2/src/parsers/caffe/binaryProtoBlob.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */

#ifndef TRT_CAFFE_PARSER_BINARY_PROTO_BLOB_H
#define TRT_CAFFE_PARSER_BINARY_PROTO_BLOB_H

#include <NvCaffeParser.h>
#include <NvInfer.h>
#include <stdlib.h>

namespace nvcaffeparser1 {
class BinaryProtoBlob : public IBinaryProtoBlob {
  public:
    BinaryProtoBlob(void *memory, nvinfer1::DataType type,
                    nvinfer1::Dims4 dimensions)
        : mMemory(memory), mDataType(type), mDimensions(dimensions) {}

    nvinfer1::Dims4 getDimensions() noexcept override { return mDimensions; }
    nvinfer1::DataType getDataType() noexcept override { return mDataType; }
    const void *getData() noexcept override { return mMemory; }
    void destroy() noexcept override { delete this; }
    ~BinaryProtoBlob() override { free(mMemory); }

    void *mMemory;
    nvinfer1::DataType mDataType;
    nvinfer1::Dims4 mDimensions;
};
} // namespace nvcaffeparser1

#endif // TRT_CAFFE_PARSER_BINARY_PROTO_BLOB_H