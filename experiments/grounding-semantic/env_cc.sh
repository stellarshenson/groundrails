# Source this before any vLLM FP8 / triton-JIT run: the container image ships no
# C compiler, and triton JIT-compiles at runtime. Toolchain lives in the
# persistent cudabuild conda env (conda-forge gcc 13.4); Python 3.12 headers come
# from the uv-managed cpython. Keep CPATH narrow - a broad CPATH breaks
# conda-gcc's glibc headers (my-gpu r9 recipe).
export PATH="/home/lab/.conda/envs/cudabuild/bin:$PATH"
export CC=/home/lab/.conda/envs/cudabuild/bin/cc
export CPATH="/home/lab/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12"
