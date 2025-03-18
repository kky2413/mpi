#include<stdio.h>

/***********************************************************************************
                         Check errors from CUDA API calls
 ***********************************************************************************/
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

/***********************************************************************************
                         Check errors from CUDA kernel calls
 ***********************************************************************************/
#define CHECK_KERNEL()                                                         \
{                                                                              \
    const cudaError_t error = cudaGetLastError();                              \
    if ( cudaSuccess != error )                                                \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
    const cudaError_t err = cudaDeviceSynchronize();                           \
    if( cudaSuccess != err )                                                   \
    {                                                                          \
        fprintf(stderr, "Synchronization Fail: %s:%d, ", __FILE__, __LINE__);  \
        fprintf(stderr, "code: %d, reason: %s\n", err,                         \
                 cudaGetErrorString(err));                                     \
        exit(1);                                                               \
    }			                                                       \
}

 /***********************************************************************************
   'cudaDeviceSynchronize()' is more careful checking which will affect performance
                             Comment out if necessary
 ***********************************************************************************/

