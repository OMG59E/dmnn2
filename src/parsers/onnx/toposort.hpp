/***
 * @Author: xingwg
 * @Date: 2024-12-12 17:01:34
 * @LastEditTime: 2024-12-12 17:01:48
 * @FilePath: /dmnn2/src/parsers/onnx/toposort.hpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>

namespace {

enum NodeState { NODE_UNVISITED, NODE_ACTIVE, NODE_VISITED };

template <class Container>
bool get_post_order(size_t node_idx, Container const &nodes,
                    std::unordered_map<std::string, size_t> const &node_map,
                    std::vector<NodeState> *node_states,
                    std::vector<size_t> *order);

}  // namespace

template <class Container>
bool toposort(Container const &nodes, std::vector<size_t> *order);