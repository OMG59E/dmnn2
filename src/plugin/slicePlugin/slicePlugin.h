/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 14:03:43
 * @FilePath: /dmnn2/src/plugin/slicePlugin/slicePlugin.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */

#ifndef ALGORITHMS_SLICEPLUGIN_H
#define ALGORITHMS_SLICEPLUGIN_H
#include <cassert>
#include <vector>
#include <NvInferPlugin.h>
#include <cublas_v2.h>
#include "../common/plugin.h"
#include "base_types.h"

namespace nvinfer1 {
    namespace plugin {
        static const char *SLICE_PLUGIN_VERSION{"1"};
        static const char *SLICE_PLUGIN_NAME{"Slice_TRT"};
        class Slice : public IPluginV2IOExt {
        public:
            Slice(int axis, int slice_dim, std::vector<int> slice_points,
                  DataType data_type, int channel_in, int height_in, int width_in)
                    : axis_{axis}, slice_dim_{slice_dim}, data_type_{data_type} {
                slice_points_.swap(slice_points);
                num_output_ = slice_points_.size() + 1;
                bottom_shape_ = nv::DimsCHW(channel_in, height_in, width_in);

                int prev = 0;
                std::vector<int> slices;
                slices.clear();
                for (int i = 0; i < slice_points_.size(); ++i) {
                    assert(slice_points_[i] > prev);
                    slices.push_back(slice_points_[i] - prev);
                    prev = slice_points_[i];
                }
                slices.push_back(bottom_shape_.d[axis_ - 1] - prev);

                top_shapes_.clear();
                nv::DimsCHW shape = bottom_shape_;
                for (int i = 0; i < num_output_; ++i) {
                    shape.d[axis_ - 1] = slices[i];
                    top_shapes_.push_back(shape);
                }
            }
            Slice(const void *data, size_t length) {
                const char *d = reinterpret_cast<const char *>(data);
                axis_ = read<int>(d);
                slice_dim_ = read<int>(d);
                num_output_ = read<int>(d);
                slice_points_.clear();
                for (int i = 0; i < num_output_ - 1; ++i)
                    slice_points_.push_back(read<int>(d));
                bottom_shape_ = read<nv::DimsCHW>(d);
                data_type_ = read<DataType>(d);
                assert(d == data + length);

                int prev = 0;
                std::vector<int> slices;
                slices.clear();
                for (int i = 0; i < slice_points_.size(); ++i) {
                    assert(slice_points_[i] > prev);
                    slices.push_back(slice_points_[i] - prev);
                    prev = slice_points_[i];
                }

                slices.push_back(bottom_shape_.d[axis_ - 1] - prev);

                top_shapes_.clear();
                nv::DimsCHW shape = bottom_shape_;
                for (int i = 0; i < num_output_; ++i) {
                    shape.d[axis_ - 1] = slices[i];
                    top_shapes_.push_back(shape);
                }
            }
            Slice() = delete;
            ~Slice() override = default;
            int getNbOutputs() const noexcept override { return num_output_; }
            Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override {
                assert(nbInputDims == 1);
                assert(index >= 0 && index < num_output_);
                return Dims3(top_shapes_[index].c(), top_shapes_[index].h(), top_shapes_[index].w());
            }
            int initialize() noexcept override { return STATUS_SUCCESS; }
            void terminate() noexcept override {}
            size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }
            int enqueue(int32_t batchSize, void const* const *inputs, void* const* outputs, void *workspace,
                    cudaStream_t stream) noexcept override;
            DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept override {
                assert(nbInputs == 1);
                assert(index >= 0 && index < num_output_);
                return inputTypes[0];
            }
            size_t getSerializationSize() const noexcept override { return sizeof(int) * (3 + num_output_ - 1) + sizeof(nv::DimsCHW) + sizeof(DataType); }
            void serialize(void *buffer) const noexcept override {
                char *d = reinterpret_cast<char *>(buffer);
                write(d, axis_);
                write(d, slice_dim_);
                write(d, num_output_);
                for (int i = 0; i < num_output_ - 1; ++i)
                    write(d, slice_points_[i]);
                write(d, bottom_shape_);
                write(d, data_type_);
                assert(d == buffer + getSerializationSize());
            }
            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const noexcept override { return false; }
            bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }
            void configurePlugin(PluginTensorDesc const* in, int32_t nbInput, PluginTensorDesc const* out, int32_t nbOutput) noexcept override {
                assert(nbInput == 1);
                assert(nbOutput == num_output_);
                assert(in[0].dims.nbDims == 3);
                assert(in[0].type == out[0].type);
                assert(in[0].format == TensorFormat::kLINEAR && out[0].format == TensorFormat::kLINEAR);
                data_type_ = in[0].type;
            }
            bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut,
                                           int32_t nbInputs, int32_t nbOutputs) const noexcept override {
                assert(nbInputs == 1 && nbOutputs == num_output_ && pos < nbInputs + nbOutputs);
                return inOut[pos].type == inOut[0].type && inOut[pos].format == TensorFormat::kLINEAR;
            }
            void detachFromContext() noexcept override {}
            const char *getPluginType() const noexcept override { return SLICE_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return SLICE_PLUGIN_VERSION; }
            void destroy() noexcept override { delete this; }
            void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) noexcept override {}
            IPluginV2IOExt *clone() const noexcept override {
                auto *plugin = new Slice(axis_, slice_dim_, slice_points_, data_type_, bottom_shape_.c(), bottom_shape_.h(), bottom_shape_.w());
                plugin->setPluginNamespace(mPluginNamespace);
                return plugin;
            }
            void setPluginNamespace(const char *pluginNamespace) noexcept override { mPluginNamespace = pluginNamespace; }
            const char *getPluginNamespace() const noexcept override { return mPluginNamespace; }

        private:
            int axis_;
            int slice_dim_;
            std::vector<int> slice_points_;
            int num_output_;
            nv::DimsCHW bottom_shape_;
            DataType data_type_{DataType::kFLOAT};
            std::vector<nv::DimsCHW> top_shapes_;
            const char *mPluginNamespace;
        };

        class SlicePluginCreator : public BaseCreator {
        public:
            SlicePluginCreator() {
                mPluginAttributes.clear();
                mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("slice_dim", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("slice_point", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("channel_in", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("height_in", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("width_in", nullptr, PluginFieldType::kINT32, 1));
                mFC.nbFields = int(mPluginAttributes.size());
                mFC.fields = mPluginAttributes.data();
                mNamespace = SLICE_PLUGIN_NAME;
            }
            ~SlicePluginCreator() override = default;
            const char *getPluginName() const noexcept override { return SLICE_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return SLICE_PLUGIN_VERSION; }
            const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }
            IPluginV2IOExt *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
                const PluginField *fields = fc->fields;
                int axis = 0;
                int slice_dim = 0;
                std::vector<int> slice_points;
                slice_points.clear();
                int channel_in = 0;
                int height_in = 0;
                int width_in = 0;
                for (int i = 0; i < fc->nbFields; i++) {
                    const char *attrName = fields[i].name;
                    if (!strcmp(attrName, "axis")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        axis = *(static_cast<const int *>(fields[i].data));
                    } else if (!strcmp(attrName, "slice_dim")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        slice_dim = *(static_cast<const int *>(fields[i].data));
                    } else if (!strcmp(attrName, "slice_point")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        int size = fields[i].length;
                        const auto *data = static_cast<const int *>(fields[i].data);
                        slice_points.clear();
                        for (int k = 0; k < size; ++k)
                            slice_points.push_back(data[k]);
                    } else if (!strcmp(attrName, "channel_in")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        channel_in = *(static_cast<const int *>(fields[i].data));
                    } else if (!strcmp(attrName, "height_in")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        height_in = *(static_cast<const int *>(fields[i].data));
                    } else if (!strcmp(attrName, "width_in")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        width_in = *(static_cast<const int *>(fields[i].data));
                    }
                }
                auto *plugin = new Slice(axis, slice_dim, slice_points, DataType::kFLOAT, channel_in, height_in, width_in);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }
            IPluginV2IOExt *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
                auto *plugin = new Slice(serialData, serialLength);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

        private:
            PluginFieldCollection mFC{};
            std::vector<PluginField> mPluginAttributes{};
        };
    }
}

#endif //ALGORITHMS_SLICEPLUGIN_H
