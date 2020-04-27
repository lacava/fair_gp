/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "error_handling.h"
#include "cuda_utils.h"
#include "cuda_utils.h"
/* #include "../node/n_2dgaussian.h" */

namespace FT{
   		
    __global__ void Gaussian2D(float * x, float x1mean, float x1var, float x2mean, float x2var, 
                               size_t idx, size_t N)
    {                    
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            x[(idx-2)*N+i] = exp(-(pow(x[(idx-1)*N+i] - x1mean, 2) / (2 * x1var) +
                                   pow(x[(idx-2)*N+i] - x2mean, 2) / (2 * x2var))); 
        }
        return;
    }
    void GPU_Gaussian2D(float * x, float x1mean, float x1var, float x2mean, float x2var, 
                         size_t idx, size_t N)
    {
        Gaussian2D<<< DIM_GRID, DIM_BLOCK >>>(x, x1mean, x1var, x2mean, x2var, idx, N);
    }
}	


