/***
 * @Author: xingwg
 * @Date: 2024-10-22 09:03:09
 * @LastEditTime: 2024-10-22 09:03:14
 * @FilePath: /dmnn2/include/utils/to_bin.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once
#include <fstream>
#include <string>

static void to_bin(void *data, size_t size, const char *filename) {
    std::ofstream fs(filename, std::ios::binary);
    std::string str = std::string(static_cast<char *>(data), size);
    fs << str;
    fs.close();
}