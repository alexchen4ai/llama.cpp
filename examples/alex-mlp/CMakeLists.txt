# main-ctx
set(TARGET alex-mlp)
add_executable(${TARGET} main-ctx.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# If we have multiple cpp code, we can create multiple executable
