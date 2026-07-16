#include <stdio.h>

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
  int indexWithinTheGrid;
  indexWithinTheGrid = blockIdx.x * blockDim.x + threadIdx.x;
  int gridStride = gridDim.x * blockDim.x;

  for (int i = indexWithinTheGrid; i < N; i += gridStride)
  {
      result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
cudaError_t err1;
cudaError_t err2;
cudaError_t err3;
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  err1=cudaMallocManaged(&a, size);
  err2=cudaMallocManaged(&b, size); 
  err3=cudaMallocManaged(&c, size);

    if (err1 != cudaSuccess)                           // `cudaSuccess` is provided by CUDA.
      printf("Error: %s\n", cudaGetErrorString(err1)); // `cudaGetErrorString` is provided by CUDA.

    if (err2 != cudaSuccess)                           // `cudaSuccess` is provided by CUDA.
      printf("Error: %s\n", cudaGetErrorString(err2)); // `cudaGetErrorString` is provided by CUDA.

if (err3 != cudaSuccess)                           // `cudaSuccess` is provided by CUDA.
      printf("Error: %s\n", cudaGetErrorString(err3)); // `cudaGetErrorString` is provided by CUDA.


cudaError_t err;
  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

    

  addVectorsInto<<<1024,128>>>(c, a, b, N);
    err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
