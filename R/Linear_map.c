/*******************************************************************************

Title  : 	Linear Network (Mapping)
Inputs : 	1. Input File Containing Training Data         (Infile)
				2. No. of Inputs                               (N)
				3. No. of outputs                              (Nout)

Outputs:    Mapping Error at each node

Notations Used:
	Nv   --> No. of patterns
	N    --> No. of inputs
	Nout --> No. of outputs
	x    --> Input Vector
	xa   --> Augmented input vector
	Wb	  --> I/P to O/P Weights			/ change: instead of Wo/
	d	  --> Output threshold           / change:  instead of to/
	ty	  --> Desired output
	y	  --> Actual output
	R	  --> Auto-correlation matrix
	C    --> Cross-correlation matrix
	dmean --> desired mean
	dstd  --> desired standard deviation

*******************************************************************************/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.1415926
#define dmean 0.0
#define dstd 0.5


//Define three seeds for random number generator
int IX=3;
int IY=4009;
int IZ=234;


void main(void)
{

	char Infile[100];
	FILE *ifs;
	int i, j;
	int Nout, N, Nv;
	float *x, *xa, *ty;
	double **R, **C;
	double **Wo, *to, **W;


	void CGrad(int ,double *, double **,double *);
	void error(int, int, char *, double **);
	float slete(float , float );

	/* Getting relevant parameters from the user */
	printf("Enter input file : ");
	gets(Infile);

	printf("Enter the No. of Inputs : ");
	scanf("%d",&N);

	printf("Enter the No. of outputs : ");
	scanf("%d",&Nout);



	/* Opening the input file in "read" mode */
	ifs= fopen(Infile,"r");


	/* Checking for the existance of the file */
	if (ifs == NULL )
	{
		perror(Infile);
		exit(1);
	}


	/* Allocating Memory dynamically*/

	x= (float *)malloc(sizeof(float)*(N));
	xa= (float *)malloc(sizeof(float)*(N+1));
	ty= (float *)malloc(sizeof(float)*Nout );
	to= (double *)malloc(sizeof(double)*Nout );

	R= (double **)malloc(sizeof(double *)*(N+1));
	C= (double **)malloc(sizeof(double *)*Nout);


	for (i=0;i<N+1;i++) {
		R[i]= (double *)malloc(sizeof(double)*(N+1));
	}

	for (i=0;i<Nout;i++){
		C[i]=(double *)malloc(sizeof(double )*(N+1));
	}

	Wo= (double **)malloc(sizeof(double *)*Nout);

	for (i=0;i<Nout;i++) {
		Wo[i]= (double *)malloc(sizeof(double )*N);
	}


	W= (double **)malloc(sizeof(double *)*Nout);

	for (i=0;i<Nout;i++) {
		W[i]= (double *)malloc(sizeof(double )*(N+1));
	}



	/* randomizing weights and thresholds using gaussian random no.
		generating subroutine "slete" */

	for (i=0;i<Nout;i++)
	{
		to[i]= slete(dstd, dmean);
		for (j=0;j<N;j++){
			Wo[i][j]=slete(dstd, dmean);
		}
	}


	for (i=0;i<Nout;i++)
	{
		for (j=0;j<N;j++) {
			W[i][j]= Wo[i][j];
		}

		W[i][N]= to[i];
	}


	for (i=0;i<N+1;i++)
	{
		for (j=0;j<N+1;j++) {
			R[i][j]=0.0;
		}

		for (j=0;j<Nout;j++){
			C[j][i]=0.0;
		}
	}


	Nv= 0;


	/* Here we determine the auto and cross correlation matrices
	by passing thro' the data once */

	while(!feof(ifs))
	{
		if(feof(ifs)) 
		{ 
			printf("\nBreaking on read(x)\n"); 
			break; 
		}
		
		for (i=0;i<N && !feof(ifs);i++) {
			fscanf(ifs,"%f",&x[i]);
		}

		if(feof(ifs)) 
		{ 
			printf("\nBreaking on read(x)\n"); 
			break; 
		}

		for (i=0;i<Nout && !feof(ifs);i++) {
			fscanf(ifs,"%f",&ty[i]);
		}
		
		for (i=0;i<N;i++){
			xa[i]= x[i];
		}
		
		/* xa - augmented input vector containing N+1 elements */
		xa[N]=1.0;

		/* Finding the Auto-correlation */

		for (i=0;i<=N;i++) {
			for (j=0;j<=N;j++) {
				R[i][j]+=xa[i]*xa[j];
			}
		}


		/*	Finding the Cross-Correlation */

		for (i=0;i<Nout;i++) {
			for (j=0;j<=N;j++) {
				C[i][j]+=ty[i]*xa[j];
			}
		}

		Nv++;
	}														/* end of while loop */
	fclose(ifs);
	// Nv--;

	/* Normalizing the correlations  */

	for (i=0;i<N+1;i++)
	{
		for (j=0;j<N+1;j++) {
			R[i][j]/=Nv;
		}

		for (j=0;j<Nout;j++) {
			C[j][i]/=Nv;
		}
	}



   /* Calling conjugate gradient subroutine */ 

	for (i=0;i<Nout;i++)
	{
		CGrad(N+1, W[i], R, C[i]); 	
	}

	/* after modifying the weights using "conjugate gradient" subroutine
		we need to call the "error" subroutine to calculate the error
		at each node */


   /* Calling subroutine error */
	error(N, Nout, Infile, W);


	printf("\n\n\n\n\n\n\n\n\nTraining complete !");

}																	/* end of main */










/******Sub-routine: Conjugate Gradient****************************************/



void CGrad(int Nu,double *w,double **r,double *c)
{
/* 	This is the program to solve for weights for single output system
	Nu: number of weights 
	*w: weight vector  
	**r: auto-correlation matrix
	*c: cross-correlation vector
*/

	double XD, XN, Num, Den, tempg, B1, B2;
	double *p, *g;
	int l, m , iter, pass;


	XD = 1;


	p= (double *)malloc(sizeof(double)*Nu);
	g= (double *)malloc(sizeof(double)*Nu);


	/* passing through the CG routine two times for improved results */
	for (pass =0;pass<2;pass++)
	{

		for(l=0;l<Nu;l++) {
			p[l] = 0.0;
			g[l] = 0.0;
		}

		for(iter= 0; iter<Nu; iter++)	  					/* start of Iteration loop */
		{
			for(l=0;l<Nu;l++) {  		  					/* start of l -loop */
				tempg = 0.0;

				for(m=0;m<Nu;m++) {
					tempg += w[m]*r[m][l];
				}
				/* gradient vector */
				g[l] = -2.0*c[l]+2.0*tempg;
			}                                         /* end of l -loop*/

			XN = 0;

			for(l=0;l<Nu;l++) {
				XN += (g[l]*g[l]);
			}

			B1 = XN/XD;
			XD = XN;

			/* direction vector */
			for(l=0;l<Nu;l++) {
				p[l] = -g[l]+B1*p[l];
			}

			Den = Num = 0.0;

			for(l=0;l<Nu;l++)										/* start of l - loop */
			{
				/* Num --> numerator of B2 */
				Num += p[l]*g[l]/-2.0;

				/* Den --> denominator of B2 */
				for(m=0;m<Nu;m++) {
					Den += p[m]*p[l]*r[m][l];
				}

			}                                   			/* end of   l - loop */

			B2 = Num/Den;


			for(l=0;l<Nu;l++) {
				w[l]+= B2*p[l];								/* Updating the weights */
			}

		}              							/* end of Iteration loop */

	}					  					  /* end of pass - loop */

}                             /* end of CjGrad */












/******** Subroutine :error ***************************************************

This subroutine calculates the mean square error at each node and prints them.

*******************************************************************************/

void error(int N2, int Nout2, char *Infile, double **W)
{

	  FILE  *ifs;
	  float *Error, y;
	  float *x, *xa, *ty;
	  int i, j;
     float MSE=0.0;
     int Nv=0;

	  /* Memory allocation */

	  x = (float *) malloc(sizeof(float)*(N2));
	  xa = (float *) malloc(sizeof(float)*(N2+1));
	  ty = (float *) malloc(sizeof(float)*(Nout2));
	  Error = (float *) malloc(sizeof(float)*(Nout2));


	  /* opening the input file in read mode */	
	  ifs= fopen(Infile,"r");


	  for (i=0;i<Nout2;i++){
		  Error[i]=0.0;
	  }

	  while(!feof(ifs))
	  {
     		Nv++;
			for(i=0;i<N2 && !feof(ifs);i++) {
				fscanf(ifs,"%f",&x[i]);
			}

			for(i=0;i<Nout2 && !feof(ifs);i++) {
				fscanf(ifs,"%f",&ty[i]);
			}

			for (i=0;i<N2;i++) {
				xa[i]= x[i];
			}

			xa[N2]=1.0;



		/*here we calculate the error for all the output nodes at the same time*/

			for (i=0;i<Nout2;i++)
			{
				y = 0.0;

				for(j=0;j<N2+1;j++) {
					y += xa[j]*W[i][j];
				}

				Error[i]+= (ty[i]-y)*(ty[i]-y);
			}

			/* y= actual output  */
			/* t= desired output */


	}
	fclose(ifs);
   Nv--;

	for (i=0;i<Nout2;i++) {
		MSE+=Error[i]/Nv;
		printf("\nError at node %d : %f",(i+1),Error[i]/Nv);
	}

	printf("\n\nTotal MSE : %f",MSE);


} 	/*End of error() */









/******************************************************************************

	Subroutine : Random no. Generator
		-the random nos. generated are uniformly distributed
		 between 0 and 1.
		-IX, IY, IZ are the SEEDS

*******************************************************************************/


float rand1(int *ix, int *iy, int *iz)
{
	int ixx, iyy, izz;
	float itemp;
	float temp;


	ixx=(*ix)/177;
	*ix=171*(*ix%177)-2*ixx;

	if(*ix < 0)
		*ix+=30269;

	iyy=(*iy)/176;
	*iy=176*(*iy%176)-2*iyy;

	if(*iy < 0)
		*iy+=30307;

	izz=(*iz)/178;
	*iz=170*(*iz%178)-2*izz;

	if(*iz < 0)
		*iz+=30323;

	temp=(float)(*ix)/30629.0+(float)(*iy)/30307.0+(float)(*iz)/30323.0;
	itemp=floor(temp);
	return (temp-itemp);

}




/********* Gaussian Random No. generator **************************************/


float slete(float std, float xmean)
{
	float rand1(int *, int *, int *);
	return (xmean+std*cos(2*PI*rand1(&IX, &IY, &IZ))*sqrt(-2.0*log(rand1(&IX, &IY, &IZ))));

}

