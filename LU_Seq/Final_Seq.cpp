//-----------------------------------------------------------------------
// Matrix Multiplication - Sequential version to run on single CPU core only
//-----------------------------------------------------------------------
//  Parallel and Distributed System (PDS) Lab
//  Updated in 8/8/2011
//-----------------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <cmath>
#include <time.h>
#include <cstdlib>
using namespace std;
//-----------------------------------------------------------------------
//   Get user input for matrix dimension or printing option
//-----------------------------------------------------------------------

typedef float** twoDPtr;

bool GetUserInput(int argc, char *argv[],int& n,int& isPrint)
{
    bool isOK = true;
    
    if(argc < 2)
    {
        cout << "Arguments:<X> [<Y>]" << endl;
        cout << "X : Matrix size [X x X]" << endl;
        cout << "Y = 1: print the input/output matrix if X < 10" << endl;
        cout << "Y <> 1 or missing: does not print the input/output matrix" << endl;
        
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
        
        //is print the input/output matrix
        if (argc >=3)
            isPrint = (atoi(argv[2])==1 && n <=9)?1:0;
        else
            isPrint = 0;
    }
    return isOK;
}

//-----------------------------------------------------------------------
//Initialize the value of matrix x[n x n]
//-----------------------------------------------------------------------
float** InitializeMatrix(int n, float value)
{
    
    // allocate square 2d matrix
    float **x = new float*[n];
    for(int i = 0 ; i < n ; i++)
        x[i] = new float[n] ;
    
    
    // assign random values
    srand (time(NULL));
    for (int i = 0 ; i < n ; i++)
    {
        for (int j = 0 ; j < n ; j++)
        {
            if (value == 1)  // generate input matrices (a and b)
                x[i][j] = (float)((rand()%10)/(float)2);
            else
                x[i][j] = 0;  // initializing resulting matrix
        }
    }
    
    return x ;
}
//------------------------------------------------------------------
//Delete matrix x[n x n]
//------------------------------------------------------------------
void DeleteMatrix(float **x,int n)
{
    
    for(int i = 0; i < n ; i++)
        delete[] x[i];
    
}
//------------------------------------------------------------------
//Print matrix
//------------------------------------------------------------------
void PrintMatrix(float** x, int n)
{
    
    for (int i = 0 ; i < n ; i++)
    {
        cout<< "Row " << (i+1) << ":\t" ;
        for (int j = 0 ; j < n ; j++)
        {
            cout << setiosflags(ios::fixed) << setprecision(2) << x[i][j] << " ";
        }
        cout << endl ;
    }
}
//------------------------------------------------------------------
//Do LU Decomposition
//------------------------------------------------------------------
void LU_Decomposition(float** matrix, float** lower, float** upper, int size)
{
    for (int i = 0; i < size; i++)
    {
        // Upper Triangular matrix
        for (int k = i; k < size; k++)
        {
            // Summation of L(i, j) * U(j, k)
            int sum = 0;
            for (int j = 0; j < i; j++)
                sum += (lower[i][j] * upper[j][k]);
            
            // Evaluating the upper trangular matrix U(i, k)
            upper[i][k] = matrix[i][k] - sum;
        }
        
        // Lower Triangular matrix
        for (int k = i; k < size; k++)
        {
            if (i == k)
                lower[i][i] = 1; // Diagonal as 1
            else
            {
                // Summation of L(k, j) * U(j, i)
                int sum = 0;
                for (int j = 0; j < i; j++)
                    sum += (lower[k][j] * upper[j][i]);
                
                // Evaluating the lower trangular matrix L(k, i)
                lower[k][i]
                = (matrix[k][i] - sum) / upper[i][i];
            }
        }
    }
}
//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[])
{
    int	n,isPrint;
    double runtime;
    
    if (GetUserInput(argc,argv,n,isPrint)==false) return 1;
    
    cout << "Starting sequential LU Decomposition" << endl;
    cout << "Matrix size = " << n << " x " << n << endl;
    
    //Initialize the value of matrix a, b, c
    float **matrix = InitializeMatrix(n, 1.0);
    float **lower = InitializeMatrix(n, 0.0);
    float **upper = InitializeMatrix(n, 0.0);
    
    //Print the input matrices
    if (isPrint==1)
    {
        cout<< "Matrix A:" << endl;
        PrintMatrix(matrix,n);
    }
    
    runtime = clock()/(double)CLOCKS_PER_SEC;
    
    LU_Decomposition(matrix,lower,upper,n);
    
    runtime = (clock()/(double)CLOCKS_PER_SEC ) - runtime;
    
    //Print the output matrix
    if (isPrint==1)
    {
        cout<< "Lower Matrix:" << endl;
        PrintMatrix(lower,n);
        cout<< "Upper Matrix:" << endl;
        PrintMatrix(upper,n);
    }
    cout<< "Program runs in " << setiosflags(ios::fixed) << setprecision(8) << runtime << " seconds\n";
    
    DeleteMatrix(matrix,n);
    DeleteMatrix(lower,n);
    DeleteMatrix(upper,n);
    
    return 0;
}
