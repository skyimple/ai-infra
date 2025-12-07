from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='fused_layernorm',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='fused_layernorm',
            sources=[
                'fused_wrapper.cpp',
                'fused_ops.cu',
            ],
            extra_compile_args={
                'cxx': ['-O2', '-std=c++14'],
                'nvcc': [
                    '-O2', 
                    '-std=c++14',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '--expt-relaxed-constexpr',
                    '--use_fast_math',
                ]
            }
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)