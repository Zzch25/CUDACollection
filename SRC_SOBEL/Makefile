#Zachary Job
#Gsyntax MF

1NAME=sobel
1SRCCU=sobel.cu sobel_kernel.cu
1OBJM=sobel.o
1PPM=result.ppm resultCU.ppm

RM=rm -f

all: toBuild test

reclean:
	-$(RM) *.o
	-$(RM) $(1NAME)
	-$(RM) $(1PPM)
	
clean:
	-$(RM) *.o

re: reclean all clean

toBuild:
	nvcc -c $(1SRCCU)
	nvcc -o $(1NAME) $(1SRCCPP:.cpp=.o) $(1OBJM)
	
test:
	./$(1NAME)