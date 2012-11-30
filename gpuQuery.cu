#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char ** argv) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("Device count %d\n", deviceCount);
  int deviceIndex;
  for (deviceIndex = 0; deviceIndex < deviceCount; deviceIndex++){

  #ifdef __cplusplus
    cudaDeviceProp deviceProperties;
  #else
    struct cudaDeviceProp deviceProperties;
  #endif

    cudaGetDeviceProperties(&deviceProperties, deviceIndex);

    if (deviceIndex == 0) {
      if (deviceProperties.major == 9999 && deviceProperties.minor == 9999) {
        printf("No CUDA GPU has been detected");
      } else if (deviceCount == 1) {
        printf("There is 1 device supporting CUDA");
      } else {
        printf("There are %d devices supporting CUDA", deviceCount);
      }
    }

    printf("Device %d name %s\n", deviceIndex, deviceProperties.name);
    printf("Computational Capabilities: %d.%d\n", deviceIndex, deviceIndex);
    printf("Maximum global memory size: %zu\n", deviceProperties.totalGlobalMem);
  }

  return 0;
}
