#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .build
tar -xzf src/llama.cpp-source.tar.gz -C .build
cmake -S .build/llama.cpp -B .build/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build .build/llama.cpp/build --config Release -j 8 --target llama-server
cp -L .build/llama.cpp/build/bin/llama-server model/runtime/
cp -L .build/llama.cpp/build/bin/lib*.so* model/runtime/
