# dmnn2

基于 TensorRT 的深度学习推理框架，支持 YOLO 系列目标检测模型的高性能推理。

## 特性

- 基于 CUDA 和 TensorRT 的高性能推理
- 支持 Caffe 和 ONNX 模型格式转换
- 支持 YOLOv5、YOLOv6、YOLO-World 等模型
- GPU 加速的图像/视频解码（FFmpeg + NVDEC）
- GPU 加速的 JPEG 编解码（nvJPEG）
- 自定义 TensorRT 插件支持
- Python 绑定支持（pybind11）

## 目录结构

```
dmnn2/
├── cmake/              # CMake 模块
├── include/            # 头文件
│   ├── codecs/         # 编解码器接口
│   ├── imgproc/        # 图像处理接口
│   ├── models/         # 模型接口
│   └── utils/          # 工具类
├── src/                # 源代码
│   ├── parsers/        # 模型解析器（Caffe/ONNX）
│   ├── plugin/         # TensorRT 自定义插件
│   └── imgproc/        # 图像处理实现
├── tools/              # 工具程序
├── samples/            # 示例代码
├── python/             # Python 绑定
├── 3rdparty/           # 第三方库
├── models/             # 模型文件目录
└── data/               # 测试数据
```

## 环境要求

| 组件 | 版本要求 |
|------|----------|
| CMake | >= 3.10 |
| CUDA | 11.1 |
| cuDNN | >= 8.x |
| TensorRT | >= 8.x |
| Protobuf | >= 3.0 |
| FFmpeg | >= 4.0 |

### 支持的 GPU 架构

- Pascal (sm_61): GTX 1060/1070/1080/1080 Ti
- Volta (sm_70): V100
- Turing (sm_75): RTX 2060/2070/2080, Tesla T4
- Ampere (sm_86): RTX 3060/3070/3080/3090, A1000-A4000

## 编译

```bash
# 创建构建目录
mkdir build && cd build

# 配置
cmake ..

# 编译
make -j$(nproc)
```

### 编译选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `CMAKE_BUILD_TYPE` | Release | 构建类型 (Release/Debug) |

## 使用示例

### 模型转换

```bash
# ONNX 转 TensorRT
./onnx2trt model.onnx -o model.engine

# Caffe 转 TensorRT
./caffe2trt model.prototxt -w model.caffemodel -o model.engine
```

### 推理示例

```bash
# YOLOv5 推理
./sample_infer_yolov5 model.engine input.jpg

# YOLOv6 推理
./sample_infer_yolov6 model.engine input.jpg
```

## 模型支持

### 目标检测

- [x] YOLOv5
- [x] YOLOv6
- [x] YOLO-World

### 自定义插件

| 插件名称 | 说明 |
|----------|------|
| YOLONMS | YOLO NMS 插件 |
| YOLONMSV2 | YOLO NMS V2 插件 |
| YOLOBox | YOLO 边界框解码插件 |
| YOLOXNMS | YOLOX NMS 插件 |
| EfficientNMS | 高效 NMS 插件 |
| Focus | Focus 层插件 |
| Slice | 切片层插件 |
| Upsample | 上采样层插件 |
| Interp | 插值层插件 |
| InstanceNormalization | 实例归一化插件 |

## 开发

### 代码风格

项目使用 Clang-Format 进行代码格式化，配置文件位于 `.clang-format`。

```bash
# 格式化代码
find . -name "*.h" -o -name "*.cpp" -o -name "*.cu" | xargs clang-format -i
```

### 代码规范

- 使用 C++17 标准
- 遵循 Google C++ 代码风格
- 命名空间: `nv::`
- 类名: PascalCase (如 `YoloV5`)
- 函数名: camelCase (如 `preprocess`)
- 变量名: snake_case (如 `batch_size`)

## 许可证

Copyright (c) 2024 by Chinasvt, All Rights Reserved.