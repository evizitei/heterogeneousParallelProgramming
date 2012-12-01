#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void cudaVectorAddition(int *dVectorA, int *dVectorB, int *dVectorC, int length){
  int dataIndex = threadIdx.x + blockDim.x * blockIdx.x;
  if(dataIndex < length){
    dVectorC[dataIndex] = dVectorA[dataIndex] + dVectorB[dataIndex];
  }
}

__host__
void vectorAddition(int *vectorA, int *vectorB, int *vectorC, int length, int size){
  //allocate device memory
  int *dVectorA;
  int *dVectorB;
  int *dVectorC;

  cudaMalloc((void **) &dVectorA, size);
  cudaMalloc((void **) &dVectorB, size);
  cudaMalloc((void **) &dVectorC, size);

  //copy host data to GPU
  cudaMemcpy(dVectorA, vectorA, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dVectorB, vectorB, size, cudaMemcpyHostToDevice);

  //let the device do the math
  int blockSize = 256;
  struct dim3 DimGrid((length - 1)/blockSize + 1, 1, 1);
  struct dim3 DimBlock(blockSize, 1, 1);
  cudaVectorAddition<<<DimGrid, DimBlock>>>(dVectorA, dVectorB, dVectorC, length);

  //copy result data back to host memory
  cudaMemcpy(vectorC, dVectorC, size, cudaMemcpyDeviceToHost);

  //free all allocated device memory
  cudaFree(dVectorA);
  cudaFree(dVectorB);
  cudaFree(dVectorC);
}

__host__
int main(int argc, char **argv){
  int vectorLength = 50;
  int vectorSize = vectorLength * sizeof(int);

  //declare vectors
  int *initialData;
  int *modulationAmount;
  int *resultData;

  //allocate memory for vectors
  initialData = (int*) malloc(vectorSize);
  modulationAmount = (int*) malloc(vectorSize);
  resultData = (int*) malloc(vectorSize);

  //create initial data
  int index;
  for(index = 0; index < vectorLength; index++){
    initialData[index] = index;
    modulationAmount[index] = index * 2;
  }

  vectorAddition(initialData, modulationAmount, resultData, vectorLength, vectorSize);

  int exampleIndex = 25;
  printf("Example element at index %d\n", exampleIndex);
  printf("initial value %d\n", initialData[exampleIndex]);
  printf("modulation amount %d\n", modulationAmount[exampleIndex]);
  printf("result data %d\n", resultData[exampleIndex]);

  //free all memory
  free(initialData);
  free(modulationAmount);
  free(resultData);
}
