#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define MAX_NUM_THREADS 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void unroll_Kernel(int Channel, int Height, int Width, int K, const float*X, float* X_unroll){
    int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q;
    int t = blockIdx.x * MAX_NUM_THREADS + threadIdx.x;
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int Width_unroll = Height_out * Width_out;
    int Height_unroll = Channel * K*K;
    
    //#define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) X[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    //#define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define X_unroll_out(i2,i1,i0) X_unroll[(i2) * (Width_unroll*Height_unroll) + (i1) * (Width_out * Height_out) + (i0)]

    if (t < (Channel * Width_unroll)){
        c = t/Width_unroll;
        s = t % Width_unroll;
        h_out = s / Width_out;
        w_out = s % Width_out;
        w_unroll = h_out * Width_out + w_out;
        w_base = c * K * K; 
        for (p=0; p <K; p++){
            for (q=0; q<K; q++){
                h_unroll = w_base + p * K +q;
                X_unroll_out(blockIdx.y, h_unroll, w_unroll) = in_4d(blockIdx.y, c, h_out +p, w_out +q);
            }
        }
    }

  //  #undef out_4d
    #undef in_4d
   // #undef mask_4d
    #undef X_unroll_out
}

__global__ void matrixMultiply(const float *A, const float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  __shared__ float subTileM[16][16];
  __shared__ float subTileN[16][16];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bz = blockIdx.z;
  int Row = by * 16.0 + ty;
  int Col = bx * 16.0 + tx;
  float Cvalue = 0;


  for (int q = 0; q < ((numAColumns-1)/16.0 +1); ++q){
    if (Row < numARows && (q*16+tx) < numAColumns)
      subTileM[ty][tx] = A[Row*numAColumns + q*16+tx];
    else
      subTileM[ty][tx] = 0;

  if (Col < numBColumns && (q*16.0+ty) < numBRows)
    subTileN[ty][tx] = B[bz*numBColumns* numBRows + (q*16+ty)* numBColumns +Col];
  else
    subTileN[ty][tx] = 0;
    __syncthreads();

  for (int k=0; k < 16; ++k)
    Cvalue += subTileM[ty][k] * subTileN[k][tx];
    __syncthreads();
  } 
if (Row < numCRows && Col < numCColumns)
  C[bz* numCColumns* numCRows + Row*numCColumns +Col] = Cvalue;
  //printf("%f",C);
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    int d_inputsize = Channel* Height*Width*Batch;
    int d_outputsize = Batch * Map_out * (Height-K+1) *(Width-K+1);
    int d_masksize = Map_out * Channel * K * K;

    cudaMalloc((void **) device_input_ptr, (d_inputsize) * sizeof(float));
    cudaMalloc((void **) device_output_ptr, (d_outputsize) * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, (d_masksize) * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, (d_inputsize)* sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, (d_masksize)* sizeof(float), cudaMemcpyHostToDevice);

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    int H_out = Height - K +1;
    int W_out = Width - K +1;
    int num_threads = Channel * H_out * W_out;
    int Width_unroll = Channel*K*K;
    int Height_unroll = H_out*W_out;
    int num_blocks = ceil((float)(Channel* H_out * W_out)/MAX_NUM_THREADS);
    float* d_X_unroll;
    cudaMalloc((void**)&d_X_unroll, Width_unroll* Height_unroll* Batch *sizeof(float));

  //  int W_size = ceil((float)Width/TILE_WIDTH); // number of horizontal tiles per output map
   // int H_size = ceil((float)Height/TILE_WIDTH); // number of vertical tiles per output map
    //int Y = H_size * W_size;
    dim3 DimNGrid(num_blocks, Batch, 1); // output tile for untiled code
    dim3 DimNBlock(MAX_NUM_THREADS, 1, 1);
    unroll_Kernel<<< DimNGrid, DimNBlock>>>(Channel, Height, Width, K, device_input, d_X_unroll);
     gpuErrchk(cudaPeekAtLastError());
    gpuErrchk( cudaDeviceSynchronize() );
    dim3 DimGrid(ceil(1.0*(H_out * W_out)/16.0), ceil(1.0*Map_out/16.0), Batch);
    dim3 DimBlock(16.0, 16.0, 1);

    matrixMultiply<<<DimGrid, DimBlock>>>(device_mask, d_X_unroll, device_output, Map_out, Width_unroll, Width_unroll, Height_unroll, Map_out, Height_unroll);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk( cudaDeviceSynchronize() );
   // cudaDeviceSynchronize();
    cudaFree(d_X_unroll);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
     int d_outputsize = Batch * Map_out * (Height-K+1) *(Width-K+1);
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, (d_outputsize * sizeof(float)), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
