#pragma once

#include "ggml.h"

struct simple_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    struct ggml_context * ctx;
};

void load_model(simple_model & model, float * a, float * b, int rows_A, int cols_A, int rows_B, int cols_B);
struct ggml_cgraph * build_graph(const simple_model& model);
struct ggml_tensor * compute(const simple_model & model);