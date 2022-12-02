#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
//#define TILE_WIDTH 32 // We will use 4 for small examples.
// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     Batch - batch_size (number of images in x)
//     Map_out - number of output feature maps
//     Channel - number of input feature maps
//     Height - input height dimension
//     Width - input width dimension
//     K - kernel height and width (K x K)
//     */

//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;
//     //(void)Height_out; // silence declared but never referenced warning. remove this line when you start working
//    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     int W_size = ceil((float)Width/TILE_WIDTH);
//     int z = blockIdx.z;
//     int m = blockIdx.x;
//     int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
//     int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;
//     //out_4d(b,m,h,w) = 0; // initialize sum to 0
//     float acc = 0.0f;

//     for (int c = 0; c < Channel; c++) { // sum over all input channels
//         for (int p = 0; p < K; p++) // loop over KxK filter
//             for (int q = 0; q < K; q++)
//                 acc += in_4d(z, c, h + p, w + q) * mask_4d(m, c, p, q);
//     }
//      if (h < Height_out && w < Width_out) {
//         out_4d(z, m, h, w) = acc;
//     }


//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }
#define TILE_WIDTH 16
__global__ void shared_tile_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    __shared__ float subtile[TILE_WIDTH + 6][TILE_WIDTH + 6];
    __shared__ float weight[7][7];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    //(void)Height_out; // silence declared but never referenced warning. remove this line when you start working
   // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_size = ceil((float)Width/TILE_WIDTH);
    int z = blockIdx.z;
    int m = blockIdx.x;
    
    int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;
    int h_base = h - threadIdx.y;
    int w_base = w - threadIdx.x;

    //out_4d(b,m,h,w) = 0; // initialize sum to 0
    float acc = 0.0f;

    // for (int c = 0; c < Channel; c++) { // sum over all input channels
    //     if((h< Height) && (w < Width)){
    //         subtile[threadIdx.y][threadIdx.x] = in_4d(z,c,h,w);
    //     }else{
    //         subtile[threadIdx.y][threadIdx.x]=0.0f;
    //     }
    // }

     for (int c = 0; c < Channel; c++) { // sum over all input channels
        if((threadIdx.x < K) && (threadIdx.y < K)){
            weight[threadIdx.y][threadIdx.x] = mask_4d(m, c, threadIdx.y, threadIdx.x);
        }
        // else{
        //     weight[threadIdx.y][threadIdx.x]=0.0f;
        // }
    
    __syncthreads();
  
        for(int j = h; j< (h_base + TILE_WIDTH + K -1); j += TILE_WIDTH){
            for (int i = w; i< (w_base + TILE_WIDTH + K -1); i += TILE_WIDTH){
                if ((i - w_base)< Width && (j-h_base) < Height){
                    subtile[j-h_base][i-w_base] = in_4d(z,c,h,w);
                }
                else{
                    subtile[j-h_base][i-w_base] = 0.0f;
                }
            }
        }

    __syncthreads();

        for (int p = 0; p < K; p++){ // loop over KxK filter
            for (int q = 0; q < K; q++)
                acc += subtile[w + p][h + q] * weight[p][q];
        }
    }
    __syncthreads();

     if (h < Height_out && w < Width_out) {
        out_4d(z, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
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
   
    int W_size = ceil((float)Width/TILE_WIDTH); // number of horizontal tiles per output map
    int H_size = ceil((float)Height/TILE_WIDTH); // number of vertical tiles per output map
    int Y = H_size * W_size;
    dim3 blockDim(TILE_WIDTH + K - 1, TILE_WIDTH + K -1, 1); // output tile for untiled code
    dim3 gridDim(Map_out, Y, Batch);
    shared_tile_kernel<<< gridDim, blockDim >>>(device_output,device_input,device_mask,Batch, Map_out, Channel,Height, Width, K);
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
