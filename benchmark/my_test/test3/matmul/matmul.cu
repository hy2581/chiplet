#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>
#include <string>

#include "apis_cu.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// 定义矩阵的宽度
#define WIDTH 1

/**
 * 矩阵乘法的核心函数，每个线程会计算结果矩阵 P 中的一个元素。
 */
__global__ void matrix_mul_gpu(int64_t *M, int64_t *N, int64_t *P, int width1)
{
    // 计算全局索引 i 和 j
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 列索引
    int j = blockIdx.y * blockDim.y + threadIdx.y; // 行索引

    int64_t sum = 0;

    // 检查索引是否在矩阵的有效范围内
    if (i < width1 * 4 && j < width1)
    {
        for (int k = 0; k < width1; k++)
        {
            int64_t a = M[j * width1 + k];
            int64_t b = N[k * width1 * 4 + i];
            sum += a * b;
        }
        P[j * width1 * 4 + i] = sum;
    }
}

int main(int argc, char **argv)
{
    // 获取当前芯粒编号
    int idX = atoi(argv[1]);
    int idY = atoi(argv[2]);

    int64_t *d_dataA, *d_dataB, *d_dataC;

    // 为矩阵 A 和 B 在设备上分配内存
    cudaMalloc((void **)&d_dataA, sizeof(int64_t) * WIDTH * WIDTH);
    cudaMalloc((void **)&d_dataB, sizeof(int64_t) * WIDTH * WIDTH * 4);
    cudaMalloc((void **)&d_dataC, sizeof(int64_t) * WIDTH * WIDTH * 4);

    // 接收来自其他芯粒的矩阵数据
    receiveMessage(idX, idY, 0, 0, d_dataA, sizeof(int64_t) * WIDTH * WIDTH);
    receiveMessage(idX, idY, 0, 0, d_dataB, sizeof(int64_t) * WIDTH * WIDTH * 4);

    // 设置线程块和网格的维度
    dim3 threadPerBlock(16, 16); // 每个 block 16x16 个线程
    dim3 blockNumber((WIDTH * 4 + threadPerBlock.x - 1) / threadPerBlock.x,
                     (WIDTH + threadPerBlock.y - 1) / threadPerBlock.y);

    // 启动矩阵乘法核函数
    // for(int i=0;i<40;i++)
    matrix_mul_gpu<<<blockNumber, threadPerBlock>>>(d_dataA, d_dataB, d_dataC, WIDTH);

    // 发送计算结果到其他芯粒
    sendMessage(0, 0, idX, idY, d_dataC, sizeof(int64_t) * WIDTH * WIDTH * 4);

    // 释放设备内存
    cudaFree(d_dataA);
    cudaFree(d_dataB);
    cudaFree(d_dataC);

    return 0;
}
