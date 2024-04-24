from setuptools import setup
from torch.utils import cpp_extension

cxx_args = []

nvcc_args = [
    '-gencode=arch=compute_86,code=sm_86',
]

setup(
    name='marlin',
    version='0.1.1',
    author='Elias Frantar',
    author_email='elias.frantar@ist.ac.at',
    description='Highly optimized FP16xINT4 CUDA matmul kernel.',
    install_requires=['numpy', 'torch'],
    packages=['marlin'],
    ext_modules=[cpp_extension.CUDAExtension(
        'marlin_cuda', ['marlin/marlin_cuda.cpp', 'marlin/marlin_cuda_kernel.cu'],
        extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
