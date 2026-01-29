#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 999999

// CPU-only computation
__host__ void random_ints(int *vec) {
    for (int i = 0; i < N; i++) {
        vec[i] = rand() % 100;
    }
}

// CPU version of add
__host__ void add_cpu(int *a, int *b, int *c) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

// GPU version of add
__global__ void add_gpu(int *a, int *b, int *c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// CPU version of print
void print_vec_cpu(int *vec) {
    for (int i = 0; i < N; i++) {
        printf("vec[%d]=%d\n", i, vec[i]);
    }
}

// GPU version of print
__global__ void print_vec_gpu(int *vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        printf("vec[%d]=%d\n", idx, vec[idx]);
    }
}

int main(void) {
    int *a, *b, *c;             // host copies
    int *d_a, *d_b, *d_c;       // device copies
    int size = N * sizeof(int); 
    int input;
    
    srand(time(NULL));  // Initialize random seed once
    
    printf("1. Compute in CPU\n2. Compute in GPU\n");
    scanf("%d", &input);
    
    // Allocate host memory
    a = (int *)malloc(size);
    random_ints(a);
    b = (int *)malloc(size);
    random_ints(b);
    c = (int *)malloc(size);
    
    if (input == 2) {  // GPU computation
        // Allocate device memory
        cudaMalloc((void **)&d_a, size);
        cudaMalloc((void **)&d_b, size);
        cudaMalloc((void **)&d_c, size);
        
        // Copy data from host to device
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
        
        // Launch add kernel on GPU
        add_gpu<<<N, 1>>>(d_a, d_b, d_c);
        
        // Copy result back from device to host
        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        
        // Or print from device (output may be unordered)
        printf("\nResults from GPU kernel:\n");
        print_vec_gpu<<<N, 1>>>(d_c);
        cudaDeviceSynchronize();
        
        // Free device memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        
    } else if (input == 1) {  // CPU computation
        printf("\nComputing on CPU...\n");
        add_cpu(a, b, c);
        
        // Print results
        printf("\nResults (computed on CPU):\n");
        print_vec_cpu(c);
        
    } else {
        printf("Invalid input!\n");
    }
    
    // Free host memory
    free(a);
    free(b);
    free(c);
    
    return 0;
}