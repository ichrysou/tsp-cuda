all:	cudastyle.cu
	nvcc cudastyle.cu -o cs -arch sm_11 -O3
