#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 2
#define INPUT_CONSTANT 7
#define BLOCK_SIZE 3


__host__
float *allocateMatrix(int rows, int columns){
  return (float *) malloc(sizeof(float) * rows * columns);
}

__host__
void initializeMatrix(int rows, int columns, float *matrix){
  float constantMultiplicant = INPUT_CONSTANT;
  int index;

  for(index = 0; index < (rows * columns); ++index){
    matrix[index] = (index+1) * constantMultiplicant;
  }
}

__host__
void naiveMatrixMultiply(float *mA, float *mB, float *mC, int aRows, int aCols, int bRows, int bCols){
  int rowIndex;
  int colIndex;
  for(rowIndex = 0; rowIndex < aRows; ++rowIndex){
    for(colIndex = 0; colIndex < bCols; ++colIndex){
      float resultValue = 0;
      int index;
      for(index = 0; index < aCols; ++index){
        float aValue = mA[(rowIndex * aCols) + index];
        float bValue = mB[(index * bCols) + colIndex];
        resultValue += (aValue * bValue);
      }
      mC[(rowIndex * bCols) + colIndex] = resultValue;
    }
  }
}


__global__
void cudaMatrixMultiply(float *mA, float *mB, float *mC, int aRows, int aCols, int bRows, int bCols){
  //currently untiled, trying to get the basics to work.
  int row = blockIdx.y * blockDim.y + threadIdx.y; /* row of matrix A to consider for this thread */
  int column = blockIdx.x * blockDim.x + threadIdx.x; /* column of matrix B to consider for this thread */

  if(row < aRows && column < bCols){
    float resultValue = 0;
    int k;

    for(k = 0; k < aCols; k++){
      float aValue = mA[(row * aCols) + k];
      float bValue = mB[(k * bCols) + column];
      resultValue += (aValue * bValue);
    }

    mC[(row * bCols) + column] = resultValue;
  }
}

__host__
void multiplyMatrices(float *mA, float *mB, float *mC, int aRows, int aCols, int bRows, int bCols){
  //declare device matrices
  float *gpuMatrixA;
  float *gpuMatrixB;
  float *gpuMatrixC;

  int aSize = aRows * aCols * sizeof(float);
  int bSize = bRows * bCols * sizeof(float);
  int cSize = aRows * bCols * sizeof(float);

  //allocate device global memory
  cudaMalloc((void **) &gpuMatrixA, aSize);
  cudaMalloc((void **) &gpuMatrixB, bSize);
  cudaMalloc((void **) &gpuMatrixC, cSize);

  //copy host data to GPU global memory
  cudaMemcpy(gpuMatrixA, mA, aSize, cudaMemcpyHostToDevice);
  cudaMemcpy(gpuMatrixB, mB, bSize, cudaMemcpyHostToDevice);

  //initialize kernal dimensions
  struct dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1); // 16 * 16 = 256 threads per block
  struct dim3 numberOfBlocks(bCols/threadsPerBlock.x + 1, aRows/threadsPerBlock.y + 1, 1); //need enough blocks to cover the full result matrix

  //delegate matrix manipulation to GPU

  //naiveMatrixMultiply(mA, mB, mC, aRows, aCols, bRows, bCols);
  cudaMatrixMultiply<<<numberOfBlocks, threadsPerBlock>>>(gpuMatrixA, gpuMatrixB, gpuMatrixC, aRows, aCols, bRows, bCols);
  cudaThreadSynchronize(); // here we wait until all threads are finished

  //copy result back to host memory
  cudaMemcpy(mC, gpuMatrixC, cSize, cudaMemcpyDeviceToHost);

  //free allocated device memory
  cudaFree(gpuMatrixA);
  cudaFree(gpuMatrixB);
  cudaFree(gpuMatrixC);
}

__host__
void printMatrix(float *matrix, int rows, int columns){
  printf("MATRIX:\n\n");
  int rowIdx;
  for(rowIdx = 0; rowIdx < rows; ++rowIdx){
    int colIdx;
    for(colIdx = 0; colIdx < columns; ++colIdx){
      printf("%f ", matrix[(rowIdx * columns) + colIdx]);
    }
    printf("\n");
  }
  printf("\n\n");
}

__host__
int main(int argc, char **argv){

  // declare host matrices
  float *baseMatrix;
  float *modulationMatrix;
  float *resultMatrix;

  //setup matrix dimensions
  int baseRows = 3;
  int baseColumns = 3;
  int modulationRows = baseColumns; // in order for this function to be defined, the column count from A must be equal to the row count of B
  int modulationColumns = 3;
  int resultRows = baseRows;
  int resultColumns = modulationColumns;

  //allocate memory for host matrices
  baseMatrix = allocateMatrix(baseRows, baseColumns);
  modulationMatrix = allocateMatrix(modulationRows, modulationColumns);
  resultMatrix = allocateMatrix(resultRows, resultColumns);

  //build in simple data
  initializeMatrix(baseRows, baseColumns, baseMatrix);
  initializeMatrix(modulationRows, modulationColumns, modulationMatrix);

  multiplyMatrices(baseMatrix, modulationMatrix, resultMatrix, baseRows, baseColumns, modulationRows, modulationColumns);

  //output results of the calculation
  printMatrix(baseMatrix, baseRows, baseColumns);
  printMatrix(modulationMatrix, modulationRows, modulationColumns);
  printMatrix(resultMatrix, baseRows, modulationColumns);

  //free host memory
  free(baseMatrix);
  free(modulationMatrix);
  free(resultMatrix);

  return 0;
}
