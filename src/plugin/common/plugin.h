/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-12-06 17:19:00
 * @FilePath: /dmnn2/src/plugin/common/plugin.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#ifndef TRT_PLUGIN_H
#define TRT_PLUGIN_H
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>

#include "error_check.h"

// Enumerator for status
typedef enum {
    STATUS_SUCCESS = 0,
    STATUS_FAILURE = 1,
    STATUS_BAD_PARAM = 2,
    STATUS_NOT_SUPPORTED = 3,
    STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

namespace nvinfer1 { 
    namespace plugin {
        static void validateRequiredAttributesExist(std::set<std::string> requiredFieldNames, PluginFieldCollection const* fc) { 
            for (int32_t i = 0; i < fc->nbFields; i++)
                requiredFieldNames.erase(fc->fields[i].name);
            if (!requiredFieldNames.empty()) {
                std::stringstream msg{};
                msg << "PluginFieldCollection missing required fields: {";
                char const* separator = "";
                for (auto const& field : requiredFieldNames) {
                    msg << separator << field;
                    separator = ", ";
                }
                msg << "}";
                std::string msg_str = msg.str();
                LOG_ERROR(msg_str.c_str());
            }
        }
    } // namespace plugin

    namespace plugin {
        class BaseCreator : public IPluginCreator {
        public:
            void setPluginNamespace(const char* libNamespace) noexcept override { mNamespace = libNamespace; }
            const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
        protected:
            std::string mNamespace;
        };

        // Write values into buffer
        template <typename T>
        void write(char*& buffer, const T& val) {
            *reinterpret_cast<T*>(buffer) = val;
            buffer += sizeof(T);
        }

        // Read values from buffer
        template <typename T>
        T read(const char*& buffer) {
            T val = *reinterpret_cast<const T*>(buffer);
            buffer += sizeof(T);
            return val;
        }
    } // namespace plugin
} // namespace nvinfer1

#endif // TRT_PLUGIN_H
