/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 09:21:32
 * @FilePath: /dmnn2/src/parsers/caffe/blobNameToTensor.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */

#ifndef TRT_CAFFE_PARSER_BLOB_NAME_TO_TENSOR_H
#define TRT_CAFFE_PARSER_BLOB_NAME_TO_TENSOR_H

#include <map>
#include <string>

#include <NvCaffeParser.h>
#include <NvInfer.h>

namespace nvcaffeparser1 {
class BlobNameToTensor : public IBlobNameToTensor {
  public:
    ~BlobNameToTensor() override = default;

    void add(const std::string &name, nvinfer1::ITensor *tensor) {
        mMap[name] = tensor;
    }
    nvinfer1::ITensor *find(const char *name) const noexcept override {
        auto p = mMap.find(name);
        if (p == mMap.end()) {
            return nullptr;
        }
        return p->second;
    }

    nvinfer1::ITensor *&operator[](const std::string &name) {
        return mMap[name];
    }

    void setTensorNames() {
        for (auto &p : mMap) {
            p.second->setName(p.first.c_str());
        }
    }

    bool isOK() { return !mError; }

  private:
    std::map<std::string, nvinfer1::ITensor *> mMap;
    bool mError{false};
};
} // namespace nvcaffeparser1
#endif // TRT_CAFFE_PARSER_BLOB_NAME_TO_TENSOR_H