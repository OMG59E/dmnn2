/***
 * @Author: xingwg
 * @Date: 2024-12-14 19:52:48
 * @LastEditTime: 2024-12-14 19:57:07
 * @FilePath: /dmnn2/src/parsers/onnx/onnx_utils.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "onnx_utils.hpp"
#include "logging.h"

bool ParseFromFile_WAR(google::protobuf::Message *msg, const char *filename) {

    std::ifstream stream(filename, std::ios::in | std::ios::binary);
    if (!stream) {
        LOG_ERROR("Could not open file: {}", std::string(filename));
        return false;
    }
    google::protobuf::io::IstreamInputStream rawInput(&stream);
    google::protobuf::io::CodedInputStream coded_input(&rawInput);
#if GOOGLE_PROTOBUF_VERSION >= 3011000
    // Starting Protobuf 3.11 accepts only single parameter.
    coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max());
#else
    // Note: This WARs the very low default size limit (64MB)
    coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                                   std::numeric_limits<int>::max() / 4);
#endif
    return msg->ParseFromCodedStream(&coded_input);
}

bool ParseFromTextFile(google::protobuf::Message *msg, const char *filename) {
    std::ifstream stream(filename, std::ios::in);
    if (!stream) {
        LOG_ERROR("Could not open file: {}", std::string(filename));
        return false;
    }
    google::protobuf::io::IstreamInputStream rawInput(&stream);
    return google::protobuf::TextFormat::Parse(&rawInput, msg);
}
