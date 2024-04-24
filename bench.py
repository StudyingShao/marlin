import sys

import numpy as np
import torch
import marlin

import time

def benchmark(f, warmup=100, iter=500):

    # create CUDA event
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(warmup):
        f()

    start_event.record()
    for i in range(iter):
        f()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event) / iter

    # # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # # we execute many benchmarks consecutively
    # time.sleep(1.)
    return elapsed_time_ms

def get_problem(m, n, k, groupsize=-1):

    print(m, n, k)

    if groupsize == -1:
        groupsize = k
    dev = torch.device('cuda:0')
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B = torch.randint(low=-2**31, high=2**31, size=(k * n // 16,), device=dev)
    # print(B.dtype) -> torch.int64
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    s = torch.zeros((k // groupsize, n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B, C, B_ref, s

def benchmark_dense(A, B, C):
    res = benchmark(lambda: torch.matmul(A, B, out=C))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / res / 10 ** 9
    }

def benchmark_quant(A, B, C, s, thread_k, thread_n, sms):
    workspace = torch.zeros(C.shape[1] // 128 * 16, device=torch.device('cuda:0'))
    res = benchmark(lambda: marlin.mul(A, B, C, s, workspace, thread_k, thread_n, sms))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 4 * B.numel() + 2 * C.numel() + 2 * s.numel()) / res / 10 ** 9
    }

# Pass the SM count for known GPUs to avoid the kernel having to query this information (this is very minor)
gpu = torch.cuda.get_device_name(0)
if 'A100' in gpu:
    SMS = 108
elif 'A10' in gpu:
    SMS = 72
elif '3090' in gpu:
    SMS = 82
elif 'A6000' in gpu:
    SMS = 84
else:
    SMS = -1

MODELS = {
    # 'ideal': [
    #     (4 * 256 * SMS, 256 * SMS)
    # ],
    'Llama7B': [
        (4096, 3 * 4096), # qkv
        # (4096, 4096), # dense
        # (4096, 11008), # fc1
        # (11008, 4096) # fc2
    ],
    # 'Llama13B': [
    #     (5120, 3 * 5120),
    #     (5120, 5120),
    #     (5120, 2 * 13568),
    #     (13568, 5120)
    # ],
    # 'Llama33B': [
    #     (6656, 3 * 6656),
    #     (6656, 6656),
    #     (6656, 2 * 17664),
    #     (17664, 6656)
    # ],
    # 'Llama65B': [
    #     (8192, 3 * 8192),
    #     (8192, 8192),
    #     (8192, 2 * 21760),
    #     (21760, 8192)
    # ],
    # 'Falcon180B': [
    #     # Note that parallel attention and FC allows layer fusions
    #     (14848, 14848 * 5 + 1024),
    #     (14848 * 5, 14848)
    # ]
}

# Set to true in order to run a more complete benchmark sweep; the default is reproduce README experiments
ALL = True

# for groupsize in [-1, 128] if ALL else [128]:
if True:
    groupsize = 128
    print('groupsize=%d' % groupsize)
    print()
    for model, layers in MODELS.items():
        print(model)
        if ALL:
            # batchsizes =  [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
            batchsizes =  [1, 2, 4, 8, 16, 32, 64, 128]
        else:
            batchsizes = [1, 2, 4, 8, 16, 32, 64, 128]
        for batch in batchsizes:
            if not ALL and model != 'ideal' and batch != 16:
                continue
            # tot_q = {'s': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0} 
            tot_q = {'s': 0, 'TFLOP/s': 0, 'GB/s': 0}
            for layer in layers:
                A, B, C, B_ref, s = get_problem(batch, layer[1], layer[0], groupsize)
                # res_d = benchmark_dense(A, B_ref, C)
                if model == 'ideal' and batch == 16:
                    # This is a special case constructed to be optimal for a thread-shape different than the default one
                    res_q = benchmark_quant(A, B, C, s, 64, 256, SMS)
                else:
                    res_q = benchmark_quant(A, B, C, s, -1, -1, SMS)
                # res_q['speedup'] = res_d['s'] / res_q['s']
                tot_q['s'] += res_q['s']
                for k in tot_q:
                    if k != 's':
                        tot_q[k] += res_q[k] * res_q['s']
            for k in tot_q:
                if k != 's':
                    tot_q[k] /= tot_q['s']
            print('batch=%04d: s=%.5f, TFLOP/s=%07.3f, GB/s=%08.3f' % (
                batch,
                tot_q['s'] * 1000,
                tot_q['TFLOP/s'],
                tot_q['GB/s']
            ))
            # print('batch=%04d: s=%.5f, TFLOP/s=%07.3f, GB/s=%08.3f, speedup=%.2f' % (
            #     batch,
            #     tot_q['s'],
            #     tot_q['TFLOP/s'],
            #     tot_q['GB/s'],
            #     tot_q['speedup']
            # ))
        print()
