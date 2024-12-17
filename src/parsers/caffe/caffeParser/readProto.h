/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 08:46:55
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/readProto.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#ifndef TRT_CAFFE_PARSER_READ_PROTO_H
#define TRT_CAFFE_PARSER_READ_PROTO_H

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "trtcaffe.pb.h"
#include <fstream>

namespace nvcaffeparser1 {
// There are some challenges associated with importing caffe models. One is that
// a .caffemodel file just consists of layers and doesn't have the specs for its
// input and output blobs.
//
// So we need to read the deploy file to get the input
bool readBinaryProto(trtcaffe::NetParameter *net, const char *file,
                     size_t bufSize) {
    if (!net) {
        LOG_ERROR("input net is null");
        return false;
    }
    if (!file) {
        LOG_ERROR("input file is null");
        return false;
    }
    using namespace google::protobuf::io;

    std::ifstream stream(file, std::ios::in | std::ios::binary);
    if (!stream) {
        LOG_ERROR("Could not open file {}", file);
        return false;
    }

    IstreamInputStream rawInput(&stream);
    CodedInputStream codedInput(&rawInput);
    codedInput.SetTotalBytesLimit(int(bufSize), -1);

    bool ok = net->ParseFromCodedStream(&codedInput);
    stream.close();

    if (!ok) {
        LOG_ERROR("Could not parse binary model file");
        return false;
    }
    return ok;
}

bool readTextProto(trtcaffe::NetParameter *net, const char *file) {
    if (!net) {
        LOG_ERROR("input net is null");
        return false;
    }
    if (!file) {
        LOG_ERROR("input file is null");
        return false;
    }
    using namespace google::protobuf::io;

    std::ifstream stream(file, std::ios::in);
    if (!stream) {
        LOG_ERROR("Could not open file {}", file);
        return false;
    }

    IstreamInputStream input(&stream);
    bool ok = google::protobuf::TextFormat::Parse(&input, net);
    stream.close();
    return ok;
}
} // namespace nvcaffeparser1

#endif // TRT_CAFFE_PARSER_READ_PROTO_H