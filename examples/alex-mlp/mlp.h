#pragma once

#include "ggml.h"
#include <string>
#include <vector>

// Define the structure for a two layer MLP
struct mlp_model {
    // Weights and biases for each layer
    struct ggml_tensor * w1;
    struct ggml_tensor * b1;
    struct ggml_tensor * w2;
    struct ggml_tensor * b2;
    struct ggml_context * ctx;
};

// Function declarations
bool load_model(const std::string & fname, mlp_model & model);
void print_tensor_stats(const char* name, struct ggml_tensor* t);
struct ggml_cgraph * build_graph(
    struct ggml_context * ctx0,
    const mlp_model & model,
    const std::vector<float> & input_data,
    struct ggml_tensor ** result_ptr);
void compute_graph(
    struct ggml_cgraph * gf,
    struct ggml_context * ctx0,
    const int n_threads,
    const char * fname_cgraph);