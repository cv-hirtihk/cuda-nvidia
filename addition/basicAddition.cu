#include <stdio.h>

// __global__ -> Will exist on device, will be called from host.
__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}

int main (void) {
    int a, b, c;                // host copies
    int *d_a, *d_b, *d_c;       // device copies
    int size = sizeof(int);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    a = 1;
    b = 2;

    // Copy data from host to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
    
    // Launch add() kernel on GPU
    add<<<1, 1>>>(d_a, d_b, d_c);

    // Copy result back from device to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);


    printf("result: %d\n", c);

    // Free allocated memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
