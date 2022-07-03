***************************************************************************************
Listed below is the commands to compile/run all the different implementation
of our code. We have modified the files given in previous labs and also
added some more file to compile and run everything properly
***************************************************************************************
1. Sequential: 
   1.1: Compile source code using: Final_Seq.cpp -o Final_Seq
   1.2: Run program with: ./Final_Seq <Matrix Size> <Print Option>

***************************************************************************************
2. OpenMP:
   2.1: Compile source code using: g++ -O -fopenmp LU-openMP.cpp -o LU-openMP
   2.2: Run program with: sbatch LU-openMP_slurm.sh <Matrix Size> <Threads> <Print Options>
	-> threads= number of threads to use

***************************************************************************************
3. MPI
   3.1: Compile source code using: mpicxx LU-MPI.cpp -o LU-MPI
   3.2: Run program with: sbatch LU-MPI_slurm.sh <Matrix Size> <Print Option>
	-> The matrix size must be a multiple of ntasks

***************************************************************************************
4. CUDA 
   4.1: Compile source code using: ssh node18 nvcc -arch=sm_30 “Path”/LU-GPU.cu -o “Path”/LU-GPU
   4.2: Run program with: sbatch LU-GPU_slurm.sh <Matrix Size> <Seed> <Print Option>