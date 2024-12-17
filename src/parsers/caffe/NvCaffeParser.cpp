/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 09:21:03
 * @FilePath: /dmnn2/src/parsers/caffe/NvCaffeParser.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "caffeParser/caffeParser.h"
#include <NvCaffeParser.h>

using namespace nvcaffeparser1;

void nvcaffeparser1::shutdownProtobufLibrary() noexcept {
    google::protobuf::ShutdownProtobufLibrary();
}
// extern "C" void* createNvCaffeParser_INTERNAL() noexcept { return
// nvcaffeparser1::createCaffeParser(); }
ICaffeParser *nvcaffeparser1::createCaffeParser() noexcept {
    return new CaffeParser;
}