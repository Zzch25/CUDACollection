/*
 *Zachary Job
 *sorbelGPU.cu
 *
 *needs binning
 */
 
#ifndef _SOBEL_KERNEL_
#define _SOBEL_KERNEL_

#include "defs.h"

/*
 SOOOOOORRy.. It got a little intense. I realized I had a bug and I
 hot fixed it last minute
 
 *
 *Applies the filter using shared memory->output buffer
 *
 */
__global__  void sorbelCU(int *imgIn, int *resOut,
							  const int imgWidth, const int imgHeight,
								const int imgWidthOffs, const int imgHeightOffs,
									const int imgSZ, const size_t thrs)
{
	__shared__ int imgSh[FILTER_SZ];
	
	int offsSh, o_l, o_u, loc, shloc,
			x, y, x_gr = 0, y_gr = 0;

	//Block/Thread indexes
	o_l		= THREADS_PER_DIM * blockIdx.x;
	x		= o_l + threadIdx.x;
	o_u		= THREADS_PER_DIM * blockIdx.y;
	y		= o_u + threadIdx.y;
	shloc	= threadIdx.x + threadIdx.y * THREADS_PER_DIM;	//reused to free up registers
				
	//Block offset in shared
	offsSh	= FILTER_BLOCK_OFFS + shloc + (threadIdx.y << 1);
	
	//Img to shared offset
	shloc 	= shloc << 1;
	loc		= o_l + o_u * imgWidth - imgWidth - 1			// shared start location offset in Img
			+ (shloc % FILTER_DIM)						 	// x offset modulo
			+ imgWidth * (shloc / FILTER_DIM); 				// y offset
	
	//Ensure valid result, the one thread to two indexes
	//Preferred if statement over computation in order to
	//keep shared data size at a multiple that optimized
	//its use
	if(shloc < FILTER_SZ - 1 && loc >= 0 && loc < imgSZ)
	{
		imgSh[shloc]		= imgIn[loc];
		imgSh[shloc + 1] 	= imgIn[loc + 1];
	}
	
	__syncthreads();
	
	//Apply filter to valid indexes, reuse shloc to save registers
	shloc = ((int)(x && y && x < imgWidthOffs && y < imgHeightOffs)) << sizeof(int);
	o_u  = offsSh - FILTER_DIM;
	o_l  = offsSh + FILTER_DIM;
	
	x_gr =			//Granularity x -> 3 above, 3 below, 3x3 grid, xy center
			/************+1**********************+2*******************+1*******************/
			/**/	(imgSh[o_u - 1] + 	/**/ (imgSh[o_u] << 1) +/**/  imgSh[o_u + 1] -	/**/
			/******************************************************************************/
			/**/		/*0*/	   		/**/  	/*ELEM*/		/**/ 	/*0*/			/**/
			/***********-1***********************-2*******************-1*******************/
			/**/    imgSh[o_l - 1]	- 	/**/ (imgSh[o_l] << 1) -/**/  imgSh[o_l + 1])	/**/
			/******************************************************************************/
			& shloc;
				
	y_gr = (0		//Granularity y -> 3 left, 3 right, 3x3 grid, xy center	 
			/***********-1**********************************************+1*****************/
			/**/ -  imgSh[o_u - 1]			/**/  /*0*/	/**/ +	imgSh[o_u + 1]			/**/
			/***********-2**********************************************+2*****************/
			/**/ - (imgSh[offsSh - 1] << 1)	/**//*OFST/	/**/ + (imgSh[offsSh + 1] << 1)	/**/
			/***********-1**********************************************+1*****************/
			/**/ - 	imgSh[o_l - 1]			/**/  /*0*/	/**/ + 	imgSh[o_l + 1])			/**/
			/******************************************************************************/
			& shloc;
	
	//Update result according to magnitude, or if border blacken
	resOut[x + y * imgWidth] = 255 * (int)((x_gr * x_gr + y_gr * y_gr) > thrs);
}

#endif