//-----------------------------------------------------------------------
// Parallel LU Decomposition : C++ OpenMP Version
// Modified Lab Code for the OpenMP Algorithm
// Modified by: Goma Niroula
// Last Updated: 11/25/2020
//-----------------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>  
using namespace std;
//-----------------------------------------------------------------------
//   Get user input of matrix dimension and printing option
//-----------------------------------------------------------------------
bool GetUserInput(int argc, char *argv[],int& n,int& numThreads,int& isPrint)
{
	bool isOK = true;

	if(argc < 3) 
	{
		cout << "Arguments:<X> <Y> [<Z>]" << endl;
		cout << "X : Matrix size [X x X]" << endl;
		cout << "Y : Number of threads" << endl;
		cout << "Z = 1: print the input/output matrix if X < 10" << endl;
		cout << "Z <> 1 or missing: does not print the input/output matrix" << endl;
		isOK = false;
	}
	else 
	{
		//get matrix size
		n = atoi(argv[1]);
		if (n <=0) 
		{
			cout << "Matrix size must be larger than 0" <<endl;
			isOK = false;
		}

		//get number of threads
		numThreads = atoi(argv[2]);
		if (numThreads <= 0)
		{	cout << "Number of threads must be larger than 0" <<endl;
			isOK = false;
		}

		//is print the input/output matrix
		if (argc >=4)
			isPrint = (atoi(argv[3])==1 && n <=9)?1:0;
		else
			isPrint = 0;
	}
	return isOK;
}

//-----------------------------------------------------------------------
//Initialize the value of matrix a[n x n] for upper triangular matrix
//-----------------------------------------------------------------------
void InitializeUpperMatrix(float** &a,int n)
{
	a = new float*[n]; 
	a[0] = new float[n*n];

	for (int i = 1; i < n; i++)	a[i] = a[i-1] + n;

	#pragma omp parallel for schedule(static) 
	for (int i = 0 ; i < n ; i++)
	{
		for (int j = 0 ; j < n ; j++)
		{	
            if (i == j) 
              a[i][j] = (((float)i+1)*((float)i+1))/(float)2;	
            else
              a[i][j] = (((float)i+1)+((float)j+1))/(float)2;
		}
	}
}
//-----------------------------------------------------------------------
//Initialize the value of matrix L[n x n] for lower triangular matrix
//-----------------------------------------------------------------------
void InitializeLowerMatrix(float** &L,int n){
	L = new float*[n];
	L[0] = new float[n*n];
	for (int i = 1; i < n; i++)	L[i] = L[i-1] + n;

	for (int j = 0 ; j < n ; j++)
	{	
		for (int i = 0 ; i < n ; i++)
		{
            if (i == j) 
              L[j][i] = 1;	
            else
              L[j][i] = 0;
		}
	}	
}
//------------------------------------------------------------------
//Delete matrix matrix a[n x n]
//------------------------------------------------------------------
void DeleteMatrix(float **a,int n)
{
	delete[] a[0];
	delete[] a; 
}
//------------------------------------------------------------------
//Print lower triangular matrix	
//------------------------------------------------------------------
void PrintMatrix(float **a, int n) 
{
	for (int i = 0 ; i < n ; i++)
	{
		cout<< "Row " << (i+1) << ":\t" ;
		for (int j = 0 ; j < n ; j++)
		{
			printf("%.2f\t", a[i][j]);
		}
		cout<<endl ;
	}
}
//------------------------------------------------------------------
//Print upper triangular matrix	
//------------------------------------------------------------------
void PrintMatrixU(float **a, int n) 
{
	for (int i = 0 ; i < n ; i++)
	{
		cout<< "Row " << (i+1) << ":\t" ;
		for (int j = 0 ; j < n ; j++)
		{
			if(j<i){
				a[i][j]=0;
			}
			printf("%.2f\t", a[i][j]);
		}
		cout<<endl ;
	}
}
//------------------------------------------------------------------
//Compute the LU Decomposition for matrix a[n x n] and L[n x n]
//------------------------------------------------------------------
bool ComputeLUDecomposition(float **a, float **L, int n)
{
	//Define the variables
	float pivot,gmax,pmax,temp;
	int  pindmax,gindmax,i,j,k;
	omp_lock_t lock;

	omp_init_lock(&lock);

	//Perform rowwise elimination
	for (k = 0 ; k < n-1 ; k++)
	{
		gmax = 0.0;

		//Find the pivot row among rows k, k+1,...n
		//Each thread works on a number of rows to find the local max value pmax
		//Then update this max local value to the global variable gmax
		#pragma omp parallel shared(a,L,gmax,gindmax) firstprivate(n,k) private(pivot,i,j,temp,pmax,pindmax)
		{
			pmax = 0.0;

			#pragma omp for schedule(dynamic) 
			for (i = k ; i < n ; i++)
			{
				temp = abs(a[i][k]);     
			
				if (temp > pmax) 
				{
					pmax = temp;
					pindmax = i;
				}
			}

			omp_set_lock(&lock);

			if (gmax < pmax)
			{
				gmax = pmax;
				gindmax = pindmax;
			}

			omp_unset_lock(&lock);
		}

		//If matrix is singular set the flag & quit
		if (gmax == 0) return false;

		//Swap rows if necessary
		if (gindmax != k)
		{
			#pragma omp parallel for shared(a) firstprivate(n,k,gindmax) private(j,temp) schedule(dynamic)
			for (j = k; j < n; j++) 
			{	
				temp = a[gindmax][j];
				a[gindmax][j] = a[k][j];
				a[k][j] = temp;
			}
		}

		//Compute the pivot
		pivot = -1.0/a[k][k];

		//Perform row reductions
		#pragma omp parallel for shared(a,L) firstprivate(pivot,n,k) private(i,j,temp) schedule(dynamic)
		for (i = k+1 ; i < n; i++)
		{
			temp = pivot*a[i][k];
			L[i][k]=((-1.0)*temp);
			for (j = k ; j < n ; j++)
			{
				a[i][j] = a[i][j] + temp*a[k][j];
			}
		}
	}

	omp_destroy_lock (&lock); 

	return true;
}
//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[])
{
	int n,numThreads,isPrintMatrix;
	float **a;
	float **L;
	double runtime;
	bool isOK;
	
	if (GetUserInput(argc,argv,n,numThreads,isPrintMatrix)==false) return 1;

	//specify number of threads created in parallel region
	omp_set_num_threads(numThreads);

	//Initialize the value of matrix A[n x n]
	InitializeUpperMatrix(a,n);
	InitializeLowerMatrix(L,n);
		
	if (isPrintMatrix) 
	{	
		cout<< "The input matrix: " << endl;
		PrintMatrix(a,n); 
	}

	runtime = omp_get_wtime();
    
	//Compute the LU decomposition for matrix a[n x n]
	isOK = ComputeLUDecomposition(a,L,n);

	runtime = omp_get_wtime() - runtime;

	if (isOK == true)
	{
		//The eliminated matrix is as below:
		if (isPrintMatrix)
		{
			cout<< "Upper Trianglular Matrix:" << endl;
			PrintMatrixU(a,n); 
			cout << "Lower Triangular Matrix:" << endl;
			PrintMatrix(L,n);
		}

		//print computing time
		cout<< "LU Decomposition runs in: "	<< setiosflags(ios::fixed) 
												<< setprecision(2)  
												<< runtime << " seconds\n";
	}
	else
	{
		cout<< "The matrix is singular" << endl;
	}
    
    // the code will run according to the number of threads specified in the arguments
    cout << "Matrix multiplication is computed using max of threads = "<< omp_get_max_threads() << " threads or cores" << endl;
    
    cout << " Matrix size  = " << n << endl;
    
	DeleteMatrix(a,n);	
	return 0;
}
