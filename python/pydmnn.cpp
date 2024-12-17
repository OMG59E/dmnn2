/***
 * @Author: xingwg
 * @Date: 2024-12-16 09:53:28
 * @LastEditTime: 2024-12-17 11:41:21
 * @FilePath: /dmnn2/python/pydmnn.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "models/net_operator.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include <algorithm>
#include <memory>
#include <string>

namespace py = pybind11;

class PyNetOperator {
public:
    PyNetOperator() = default;
    ~PyNetOperator() = default;
    // TODO
private:
    std::shared_ptr<nv::NetOperator> net_{nullptr};
};

PYBIND11_MODULE(pydmnn, m) {
    py::class_<PyNetOperator>(m, "NetOperator").def(py::init<>());
}