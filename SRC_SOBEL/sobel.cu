/*
 *Zachary Job
 *
 *PMM and gold donors from Prof. Mordohai
 *
 *sobel.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "string.h"
#include "defs.h"
#include "sobel_kernel.cu"

#define DEFAULT_THRESHOLD  		4000
#define DEFAULT_FILENAME 		"BWstop-sign.ppm"
#define DEFAULT_OUTPUT 			"result.ppm"
#define CUKERNEL_OUTPUT 		"resultCU.ppm"
//#define CUDEBUG_OUTPUT 		"CUDEBUG.ppm"
//#define DEDEBUG_OUTPUT 		"DEDEBUG.ppm"

/*
 * Reads a PMM into a buffer
 */
unsigned int *read_ppm( char *filename, int * xsize, int * ysize, int *maxval )
{
	//V A R I A B L E S//////////////////////////////////////////////////////////////////////////////
	
	char chars[1024];
	int num, bufsize, pixels, i;
	unsigned int width, height, maxvalue;
	long offset, numread;
	
	char duh[80], *line, *ptr;
	unsigned char *buf;
	unsigned int *pic;
	
	FILE *fp;
	
	//C O N F I G U R A T I O N   A N D   I N S T A T A N T I A T I O N//////////////////////////////
	
	if ( !filename || filename[0] == '\0')
	{
		fprintf(stderr, "read_ppm but no file name\n");
		return NULL;  // fail
	}

	fprintf(stderr, "read_ppm( %s )\n", filename);
	fp = fopen( filename, "rb");
	if (!fp) 
	{
		fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
		return NULL; // fail 
	}
	
	//E X E C U T I O N//////////////////////////////////////////////////////////////////////////////
	
	num = fread(chars, sizeof(char), 1000, fp);
	if (chars[0] != 'P' || chars[1] != '6') 
	{
		fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
		return NULL;
	}

	ptr = chars+3; // P 6 newline
	if (*ptr == '#') // comment line! 
		ptr = 1 + strstr(ptr, "\n");

	num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
	fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);  
	*xsize = width;
	*ysize = height;
	*maxval = maxvalue;
	line = chars;
  
	pic = (unsigned int *)malloc( width * height * sizeof(unsigned int));
	if (!pic) {
		fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
		return NULL; // fail but return
	}

	// allocate buffer to read the rest of the file into
	bufsize = 3 * width * height * sizeof(unsigned char);
	if ((*maxval) > 255) bufsize *= 2;
	buf = (unsigned char *)malloc( bufsize );
	if (!buf) 
	{
		fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
		return NULL; // fail but return
	}

	// find the start of the pixel data. 
	sprintf(duh, "%d\0", *xsize);
	line = strstr(line, duh);
	line += strlen(duh) + 1;

	sprintf(duh, "%d\0", *ysize);
	line = strstr(line, duh);
	line += strlen(duh) + 1;

	sprintf(duh, "%d\0", *maxval);
	line = strstr(line, duh);
	
	fprintf(stderr, "%s found at offset %ld\n", duh, line - chars);
	line += strlen(duh) + 1;

	offset = line - chars;
	fseek(fp, offset, SEEK_SET); // move to the correct offset
	numread = fread(buf, sizeof(char), bufsize, fp);
	fprintf(stderr, "Texture %s   read %ld of %d bytes\n", filename, numread, bufsize); 
	
	for (i=0, pixels = (*xsize) * (*ysize); i<pixels; i++) 
		pic[i] = (int) buf[3*i];  // red channel
	
	//C L E A N U P//////////////////////////////////////////////////////////////////////////////////
	
	fclose(fp);
	return pic; // success
}

/*
 * Writes a PMM from a buffer
 */
void write_ppm( char *filename, int xsize, int ysize, int maxval, int *pic) 
{
	//V A R I A B L E S//////////////////////////////////////////////////////////////////////////////
	
	unsigned char uc;
	int numpix, i;
	
	FILE *fp;
	
	//C O N F I G U R A T I O N   A N D   I N S T A T A N T I A T I O N//////////////////////////////
	
	fp = fopen(filename, "wb");
	if (!fp) 
	{
		fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n",filename);
		exit(-1); 
	}
  
	//E X E C U T I O N//////////////////////////////////////////////////////////////////////////////
  
	fprintf(fp, "P6\n"); 
	fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);
  
	numpix = xsize * ysize;
	for (i=0; i<numpix; i++)
	{
		uc = (unsigned char) pic[i];
		fprintf(fp, "%c%c%c", uc, uc, uc); 
	}

	//C L E A N U P//////////////////////////////////////////////////////////////////////////////////
	
	fclose(fp);
}

/*
 * Executes a serial Sobel along and launches a CUDA version
 *
 * Designed to coalesce and avoid all control statements.
 * This allows for lots of math tweaks to have a 0 divergence
 * kernel.
 */
int main( int argc, char **argv )
{
	
	//V A R I A B L E S//////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	
	int xsize, xsizeOff, ysize, ysizeOff, imgSZ, maxval, numbytes,
		i, j, magnitude, sum1, sum2, col, row, offset;
		
	int *result, *resultCU,
		*device_return_array, *device_array,
		*out;
	unsigned int *pic;
	
	int thresh = DEFAULT_THRESHOLD;
	char *filename = strdup(DEFAULT_FILENAME);
	
	//C O N F I G U R A T I O N   A N D   I N S T A T A N T I A T I O N//////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
  
	if (argc > 1)
	{  
		if (argc == 3)
			filename = strdup( argv[1]);
		thresh = atoi(argv[argc - 1]);
		
		fprintf(stderr, "file %s    threshold %d\n", filename, thresh); 
	}
	
	pic = read_ppm( filename, &xsize, &ysize, &maxval );
	imgSZ	 	= xsize * ysize;
	numbytes 	= imgSZ * 3 * sizeof( unsigned int );
	xsizeOff 	= xsize - 1;
	ysizeOff 	= ysize - 1;
	
	result = (int *) malloc( numbytes );
	if (!result)
		fprintf(stderr, "Sobel gold array attempt: unable to malloc %d bytes\n", numbytes), exit(-1); // fail
	
	resultCU = (int *) malloc( numbytes );
	if (!resultCU) 
		fprintf(stderr, "Sobel CU array attempt: unable to malloc %d bytes\n", numbytes), exit(-1); // fail
	
	cudaMalloc((void**)&device_array, numbytes / 3); //for some awesome efficency
	cudaMalloc((void**)&device_return_array, numbytes);
	
	// if memory allocation failed, report an error message
	if(device_array == 0 || device_return_array == 0)
	{
		fprintf(stderr, "couldn't allocate memory\n");
		return -1;
	}
	
	//set device to local memory
	cudaMemcpy(device_array, (int *)pic, numbytes / 3, cudaMemcpyHostToDevice);
	// Setup the execution configuration
	dim3 threads(THREADS_PER_DIM, THREADS_PER_DIM, 1);
	dim3 blocks((xsize + THREADS_PER_DIM - 1)/THREADS_PER_DIM, 
					(ysize + THREADS_PER_DIM - 1)/THREADS_PER_DIM, 1);
					
	//E X E C U T I O N//////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////

	fprintf(stderr, "sobel gold start\n");
	out = result;

	for (col=0; col<ysize; col++)
		for (row=0; row<xsize; row++,*out++ = 0); 
			
	for (i = 1;  i < ysize - 1; i++)
	{
		for (j = 1; j < xsize -1; j++)
		{
			offset = i*xsize + j;

			sum1 =  	(pic[ xsize * (i-1) + j+1 ] -     pic[ xsize*(i-1) + j-1 ] 
				+ 2 * 	 pic[ xsize * (i)   + j+1 ] - 2 * pic[ xsize*(i)   + j-1 ]
				+     	 pic[ xsize * (i+1) + j+1 ] -     pic[ xsize*(i+1) + j-1 ]);
      
			sum2 = 		(pic[ xsize * (i-1) + j-1 ] + 2 * pic[ xsize * (i-1) + j ]
				- 2 * 	 pic[ xsize * (i+1) + j ] - pic[ xsize * (i+1) + j+1 ]
				+ 		 pic[ xsize * (i-1) + j+1 ] - pic[xsize * (i+1) + j-1 ]);
				
      
			magnitude = sum1*sum1 + sum2*sum2;
			result[offset] = (magnitude > thresh) ? 255 : 0;
		}
	}

	fprintf(stderr, "sobel gold done\nsobel CU start\n");
	write_ppm(strdup(DEFAULT_OUTPUT), xsize, ysize, 255, result);	
 
    // Launch the device computation threads!
	sorbelCU<<<blocks,threads>>>(device_array, device_return_array, 
									xsize, ysize, xsizeOff, ysizeOff, 
										imgSZ, thresh);
	cudaMemcpy(resultCU, device_return_array, numbytes, cudaMemcpyDeviceToHost);
	
	fprintf(stderr, "sobel CU done\n");
	write_ppm(strdup(CUKERNEL_OUTPUT), xsize, ysize, 255, resultCU);
	
	//C L E A N U P//////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	
	free(pic);
	free(result);
	free(resultCU);
	//cudaFree(device_array); //no need
	cudaFree(device_return_array);
}