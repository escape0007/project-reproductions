# Local OpenMP runtime

`libomp.dylib` was copied, without modification, from the already installed
PyTorch package at
`/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/lib/libomp.dylib`.
It is an arm64 LLVM OpenMP runtime, used through `DYLD_LIBRARY_PATH` by `run_all.py`.
No system-wide library or linker configuration was modified.

SHA-256: `fe0670951c85401421f6cedc91971f28dc870bad4c1f0cfebe2b74dac421e5f0`.

For another machine, use that machine's OpenMP runtime. See the
[LightGBM installation guide](https://lightgbm.readthedocs.io/en/v4.6.0/Installation-Guide.html)
and [LLVM OpenMP license](https://github.com/llvm/llvm-project/blob/main/openmp/LICENSE.TXT).
