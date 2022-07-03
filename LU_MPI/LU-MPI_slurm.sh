#!/bin/bash
##############################################################################
# FILE: mpi_slurm - Intel MPI job
# DESCRIPTION:
#   Simple Slurm job command script for MPI code on Heracles
# AUTHOR: Manh and Iris
# LAST REVISED: June/2019
##############################################################################
##### These lines are for Slurm
#SBATCH --partition=day-long-cpu        ### Partition
## PartitionName=day-long-cpu Priority=20000 Default=NO MaxTime=1-0:00:00 State=UP Nodes=node[2-17]
#SBATCH --job-name=mpicode	 ### Job Name
#SBATCH --output=slurm_output.%j ### File in which to store job output
#SBATCH --error=slurm_error.%j   ### File in which to store job error messages
#SBATCH --time=0-00:01:00        ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --ntasks=24               ## -n corresponds to MPI ranks/ each rank is one task
#SBATCH --ntasks-per-node=24     ### Nuber of tasks to be launched per Node, default = 1
## SBATCH --mail-type=ALL          ### email alert at start, end and abortion of execution
## SBATCH --mail-user=myemail ### send mail to this address (
## Run the code
mpirun -print-rank-map ./LU-MPI "$@"
### Display some diagnostic information
echo '=====================JOB DIAGNOTICS========================'
date
echo -n 'This machine is ';hostname
echo -n 'My jobid is '; echo $SLURM_JOBID
echo 'My path is:' 
echo $PATH
echo 'My job info:'
squeue -j $SLURM_JOBID
echo 'Machine info'
sinfo -s
echo '========================ALL DONE==========================='
