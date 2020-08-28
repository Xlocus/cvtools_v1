#!/usr/bin/env python
import os
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def make_cuda_ext(name, module, sources, sources_cuda):

    define_macros = list()
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


if __name__ == '__main__':
    setup(
        name='cvtools',
        packages=find_packages(),
        # package_data={'cvtools.ops': ['*/*.so']},
        ext_modules=[
            make_cuda_ext(
                name='compiling_info',
                module='cvtools.ops.utils',
                sources=['src/compiling_info.cpp'],
                sources_cuda=[]),
            make_cuda_ext(
                name='nms_ext',
                module='cvtools.ops.nms',
                sources=['src/nms_ext.cpp', 'src/cpu/nms_cpu.cpp'],
                sources_cuda=[
                    'src/cuda/nms_cuda.cpp', 'src/cuda/nms_kernel.cu'
                ]),
            make_cuda_ext(
                name='roi_align_ext',
                module='cvtools.ops.roi_align',
                sources=[
                    'src/roi_align_ext.cpp',
                    'src/cpu/roi_align_v2.cpp',
                ],
                sources_cuda=[
                    'src/cuda/roi_align_kernel.cu',
                    'src/cuda/roi_align_kernel_v2.cu'
                ]),
            make_cuda_ext(
                name='roi_pool_ext',
                module='cvtools.ops.roi_pool',
                sources=['src/roi_pool_ext.cpp'],
                sources_cuda=['src/cuda/roi_pool_kernel.cu']),
            make_cuda_ext(
                name='deform_conv_ext',
                module='cvtools.ops.dcn',
                sources=['src/deform_conv_ext.cpp'],
                sources_cuda=[
                    'src/cuda/deform_conv_cuda.cpp',
                    'src/cuda/deform_conv_cuda_kernel.cu'
                ]),
            make_cuda_ext(
                name='deform_pool_ext',
                module='cvtools.ops.dcn',
                sources=['src/deform_pool_ext.cpp'],
                sources_cuda=[
                    'src/cuda/deform_pool_cuda.cpp',
                    'src/cuda/deform_pool_cuda_kernel.cu'
                ]),
            make_cuda_ext(
                name='sigmoid_focal_loss_ext',
                module='cvtools.ops.sigmoid_focal_loss',
                sources=['src/sigmoid_focal_loss_ext.cpp'],
                sources_cuda=['src/cuda/sigmoid_focal_loss_cuda.cu']),
            make_cuda_ext(
                name='masked_conv2d_ext',
                module='cvtools.ops.masked_conv',
                sources=['src/masked_conv2d_ext.cpp'],
                sources_cuda=[
                    'src/cuda/masked_conv2d_cuda.cpp',
                    'src/cuda/masked_conv2d_kernel.cu'
                ]),
            make_cuda_ext(
                name='carafe_ext',
                module='cvtools.ops.carafe',
                sources=['src/carafe_ext.cpp'],
                sources_cuda=[
                    'src/cuda/carafe_cuda.cpp',
                    'src/cuda/carafe_cuda_kernel.cu'
                ]),
            make_cuda_ext(
                name='carafe_naive_ext',
                module='cvtools.ops.carafe',
                sources=['src/carafe_naive_ext.cpp'],
                sources_cuda=[
                    'src/cuda/carafe_naive_cuda.cpp',
                    'src/cuda/carafe_naive_cuda_kernel.cu'
                ]),
            make_cuda_ext(
                name='corner_pool_ext',
                module='cvtools.ops.corner_pool',
                sources=['src/corner_pool.cpp'],
                sources_cuda=[]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
