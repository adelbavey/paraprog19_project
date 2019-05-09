//
// Gaussian mixture model
//

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_randist.h>

#include <mpi.h>

#include "parallel.c"

// global constants, feel free to configure
//
// how many components to try to fit onto the data
const unsigned int NUM_COMPONENTS = 15;
unsigned int NUM_DIMS = 0;
unsigned int NUM_DATA = 0;

const char input_name[] = "europediff.txt";

// our rank and the toal number of processors as global variables
int rank, num_processors;

//Get number of data points and dimension of points, write to NUM_DATA and NUM_DIMS
void get_N_D(void){
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    size_t read;

    fp = fopen(input_name, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    int num_infile_rows = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        num_infile_rows++;
        //printf("Retrieved line of length %zu:\n", read);
        //printf("%s", line);
        char scroll[100];
        strcpy(scroll, line);
        char *p = scroll;

        //printf (" initials : %c", *p);

        int  i = 0;
        while ((p = strpbrk (p, " "))){
                if(isdigit(*(++p)))i++;
                //*++p;
                //printf ("%c", *++p);
                }

        //printf ("\n\n");
        NUM_DIMS = i;
    }
    NUM_DATA = num_infile_rows;

    fclose(fp);
    if (line)
        free(line);
    //exit(EXIT_SUCCESS);
}

// utitilities
void print_vector(gsl_vector* vec) {
    int i;
    for (i=0; i<vec->size; ++i) {
        printf("%f ", gsl_vector_get(vec, i));
    }
    printf("\n");
}

void print_matrix(gsl_matrix* mat) {
    int i,j;
    for (i=0; i<mat->size1; ++i) {
        for (j=0; j<mat->size2; ++j) {
            printf("%f ", gsl_matrix_get(mat, i, j));
        }
        printf("\n");
    }
}

//Gaussian pdf, not used currently
double mvg(const gsl_matrix* x, const gsl_matrix* mean,const gsl_matrix* sigma){

        //Get LU decomp of sigma
        gsl_matrix* sigmaLU = gsl_matrix_alloc(NUM_DIMS,NUM_DIMS);
        gsl_matrix_memcpy(sigmaLU, sigma);
        gsl_permutation* p = gsl_permutation_alloc(NUM_DIMS);
        int signum = 0;
        gsl_linalg_LU_decomp(sigmaLU,p,&signum);

        //det of sigma
        double detSigma = gsl_linalg_LU_det(sigmaLU,signum);

        //Inverse of sigma
        gsl_matrix* invSigma = gsl_matrix_alloc(NUM_DIMS,NUM_DIMS);
        gsl_linalg_LU_invert(sigmaLU,p,invSigma);

        //x without mean
        gsl_matrix* x_nomean = gsl_matrix_alloc(1, NUM_DIMS);
        gsl_matrix_memcpy(x_nomean,x);
        gsl_matrix_sub(x_nomean,mean);

        //Calculate exponent value in mvg
        gsl_matrix* result_matrix1 = gsl_matrix_alloc(NUM_DIMS,1);
        gsl_matrix* result_matrix2 = gsl_matrix_alloc(1,1);
        gsl_blas_dgemm(CblasNoTrans,CblasTrans,1,invSigma,x_nomean,0,result_matrix1);
        gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1,x_nomean,result_matrix1,0,result_matrix2);
        double exponent = (-0.5)*gsl_matrix_get(result_matrix2,0,0);

        //Calculate final result
        double final_result = (1/sqrt( pow(2*M_PI,NUM_DIMS) * detSigma))*exp(exponent);
        gsl_matrix_free(sigmaLU);
        gsl_permutation_free(p);
        gsl_matrix_free(invSigma);
        gsl_matrix_free(x_nomean);
        gsl_matrix_free(result_matrix1);
        gsl_matrix_free(result_matrix2);
        return final_result;



}

//Initialize parameters
void initialize_components(const gsl_matrix* data, 
gsl_matrix* priors, gsl_matrix* means, gsl_matrix** sigmas){

    //Initialize priors
    //gsl_vector * priors = gsl_vector_alloc(NUM_COMPONENTS);
    gsl_matrix_set_all(priors, 1.0/NUM_COMPONENTS);

    unsigned int randint = 0;

    //Init means (non random)
    //gsl_matrix * means = gsl_matrix_alloc(NUM_COMPONENTS,NUM_DIMS);
    for (int i = 0; i < NUM_COMPONENTS; i++)
    {   
        int r = rand_r(&randint)%NUM_DATA;
        gsl_vector_const_view x_view = gsl_matrix_const_row(data,r);
        gsl_matrix_set_row(means,i,&x_view.vector);
    }

    //Init covariance matrices.
    //Create initital sigma shared by all components
    //gsl_matrix* sigmas[NUM_COMPONENTS];

    gsl_matrix* init_sigma = gsl_matrix_alloc(NUM_DIMS,NUM_DIMS);
    gsl_matrix_set_all(init_sigma,0);
    /*
    for (int i = 0; i < NUM_DIMS; i++)
    {
        for (int j = 0; j < NUM_DIMS; j++)
        {
            double cov = ((NUM_DATA-1.0)/NUM_DATA)*gsl_stats_covariance(gsl_matrix_const_column(data,i).vector.data,1,gsl_matrix_const_column(data,j).vector.data,1,NUM_DATA);
            //cov/= NUM_COMPONENTS*10;
            gsl_matrix_set(init_sigma,i,j,cov);
        }
    }
    */

   
   //Matrix with one row and n cols, filled with 1
    gsl_matrix* ones = gsl_matrix_alloc(1,NUM_DATA);
    gsl_matrix_set_all(ones, 1);

    //sample mean
    gsl_matrix* sample_mean = gsl_matrix_alloc(1,NUM_DIMS);
    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1,ones,data,0,sample_mean);
    gsl_matrix_scale(sample_mean,1.0/NUM_DATA);

    gsl_matrix_free(ones);

    //sum individual vars
    for (int data_i = 0; data_i < NUM_DATA; data_i++)
    {
        gsl_matrix_const_view data_point_view = gsl_matrix_const_submatrix(data,data_i,0,1,NUM_DIMS);

        gsl_matrix* data_point_nomean = gsl_matrix_alloc(1,NUM_DIMS);
        gsl_matrix_memcpy(data_point_nomean,&data_point_view.matrix);
        gsl_matrix_sub(data_point_nomean,sample_mean);

        gsl_matrix* result = gsl_matrix_alloc(NUM_DIMS,NUM_DIMS);
        gsl_blas_dgemm(CblasTrans,CblasNoTrans,1,data_point_nomean,data_point_nomean,0,result);

        gsl_matrix_add(init_sigma,result);

        gsl_matrix_free(data_point_nomean);
        gsl_matrix_free(result);
    }
    gsl_matrix_scale(init_sigma,1.0/NUM_DATA);
    
    
    for (int i = 0; i < NUM_COMPONENTS; i++)
    {
        sigmas[i] = gsl_matrix_alloc(NUM_DIMS,NUM_DIMS);
        gsl_matrix_memcpy(sigmas[i],init_sigma);
    }


}

//Calculate new posteriors with current parameters
void expectation_step(const gsl_matrix* data, 
const gsl_matrix* priors, const gsl_matrix* means, gsl_matrix** sigmas,
gsl_matrix* posteriors){

    //gsl_vector * work = gsl_vector_alloc(NUM_DIMS);
    double result = 0;

    for (int data_i = 0; data_i < NUM_DATA; data_i++)
    {
        double row_sum = 0;
        for (int component_i = 0; component_i < NUM_COMPONENTS; component_i++)
        {
            gsl_matrix_const_view x_view = (gsl_matrix_const_submatrix(data,data_i,0,1,NUM_DIMS));
            gsl_matrix_const_view mean_view = (gsl_matrix_const_submatrix(means,component_i,0,1,NUM_DIMS));
            //gsl_vector_const_view x_view = (gsl_matrix_const_row(data,data_i));
            //gsl_vector_const_view mean_view = (gsl_matrix_const_row(data,component_i));
            const gsl_matrix* sigma = sigmas[component_i];

            //Cholesky
            gsl_matrix * sigma_chol = gsl_matrix_alloc(NUM_DIMS,NUM_DIMS);
            gsl_matrix_memcpy(sigma_chol,sigma);
            //print_matrix(sigma);
            gsl_linalg_cholesky_decomp1(sigma_chol);

            //Get pdf value
            //gsl_ran_multivariate_gaussian_pdf(&x_view.vector,&mean_view.vector,sigma_chol,&result,work);
            result = mvg(&x_view.matrix,&mean_view.matrix,sigma);
            //printf("%f",result);

            gsl_matrix_set(posteriors,data_i,component_i,gsl_matrix_get(priors,0,component_i)*result);
            row_sum += gsl_matrix_get(priors,0,component_i)*result;

            gsl_matrix_free(sigma_chol);
        }

        gsl_vector_view posterior_row = gsl_matrix_row(posteriors,data_i);
        gsl_vector_scale(&posterior_row.vector,1.0/row_sum);

        
    }
    

}

//Calculate new parameters with current posteriors
void maximization_step(const gsl_matrix* data, 
gsl_matrix* priors, gsl_matrix* means, gsl_matrix** sigmas,
const gsl_matrix* posteriors){

    //Matrix with one row and n cols, filled with 1
    gsl_matrix* ones = gsl_matrix_alloc(1,NUM_DATA);
    gsl_matrix_set_all(ones, 1);

    //new priors
    gsl_matrix* new_priors = gsl_matrix_alloc(1,NUM_COMPONENTS);
    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1,ones,posteriors,0,new_priors);
    gsl_matrix_scale(new_priors,1.0/NUM_DATA);
    gsl_matrix_memcpy(priors,new_priors);

    //new means

    //matrix product of posterior and data
    gsl_matrix* new_means = gsl_matrix_alloc(NUM_COMPONENTS,NUM_DIMS);
    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1,posteriors,data,0,new_means);

    //divide means by summed component posteriors
    gsl_matrix* mean_divisor = gsl_matrix_alloc(NUM_COMPONENTS,1);
    gsl_blas_dgemm(CblasTrans,CblasTrans,1,posteriors,ones,0,mean_divisor);
    for (int i = 0; i < NUM_DIMS; i++)
    {
        gsl_matrix_view view = gsl_matrix_submatrix(new_means,0,i,NUM_COMPONENTS,1);
        gsl_matrix_div_elements(&view.matrix,mean_divisor);
    }
    gsl_matrix_memcpy(means,new_means);

    //new sigmas
    gsl_matrix* new_sigmas[NUM_COMPONENTS];
    for (int component_i = 0; component_i < NUM_COMPONENTS; component_i++)
    {
        new_sigmas[component_i] = gsl_matrix_alloc(NUM_DIMS,NUM_DIMS);
        gsl_matrix_set_all(new_sigmas[component_i],0);

        double posterior_sum = 0;

        /*
        // parallelise this
        for (int data_i = 0; data_i < NUM_DATA; data_i++)
        {
            gsl_matrix_const_view x_view = gsl_matrix_const_submatrix(data,data_i,0,1,NUM_DIMS);
            gsl_matrix_const_view u_view = gsl_matrix_const_submatrix(means,component_i,0,1,NUM_DIMS);
            gsl_matrix* x_cpy = gsl_matrix_alloc(1,NUM_DIMS);
            gsl_matrix_memcpy(x_cpy,&x_view.matrix);
            gsl_matrix_sub(x_cpy,&u_view.matrix);

            gsl_matrix* xx = gsl_matrix_alloc(NUM_DIMS,NUM_DIMS);
            gsl_blas_dgemm(CblasTrans,CblasNoTrans, gsl_matrix_get(posteriors,data_i,component_i) ,x_cpy,x_cpy,0,xx);
            //gsl_matrix_scale(xx,gsl_matrix_get(posteriors,data_i,component_i));

            gsl_matrix_add(new_sigmas[component_i],xx);
            posterior_sum += gsl_matrix_get(posteriors,data_i,component_i);
        }
        */
        posterior_sum = parallel_sum(component_i, data, means, posteriors, new_sigmas[component_i]);

        gsl_matrix_scale(new_sigmas[component_i],1.0/posterior_sum);
        
        gsl_matrix_memcpy(sigmas[component_i],new_sigmas[component_i]);
        gsl_matrix_free(new_sigmas[component_i]);
    }

    gsl_matrix_free(ones);
    gsl_matrix_free(mean_divisor);

    
}


int main(int argl, char* argv[]){

    int rc;

    // initialise random number generator
    //srandom(time(NULL));
    
    // initialise and setup MPI
    rc = MPI_Init(&argl, &argv);
    rc = MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Get number of data points and dims
    get_N_D();

    //Get data
    FILE * fp = fopen(input_name, "r");
    gsl_matrix * data = gsl_matrix_alloc (NUM_DATA, NUM_DIMS);
    gsl_matrix_fscanf(fp, data);

    //Allocate component memory and initialize
    gsl_matrix * priors = gsl_matrix_alloc(1,NUM_COMPONENTS);
    
    gsl_matrix * means = gsl_matrix_alloc(NUM_COMPONENTS,NUM_DIMS);
    gsl_matrix* sigmas[NUM_COMPONENTS];
    gsl_matrix* posteriors = gsl_matrix_alloc(NUM_DATA,NUM_COMPONENTS);
    initialize_components(data,priors,means,sigmas);
    //print_matrix(means);

    if (rank == 0) {
        printf("Read %d data points from file \"%s\". Starting calculations with %d components.\n",
                    NUM_DATA, input_name, NUM_COMPONENTS);
        //fflush(stdout);
    }
    //print_matrix(sigmas[0]);
    //EM ALGORITHM
    for (int iter = 0; iter < 30; iter++)
    {
        if (rank == 0) {
            printf("step: %d\n", iter);
        }
        //print_matrix(means);
        //printf("\n");
        expectation_step(data,priors,means,sigmas,posteriors);
        //print_matrix(posteriors);
        maximization_step(data,priors,means,sigmas,posteriors);
        //print_matrix(priors);
    }

    // only output if we're the root process
    if (rank == 0) {
        // output to gmm_out.txt
        FILE* outfile = fopen("gmm_out.txt", "w");

        for (int i = 0; i<NUM_COMPONENTS; ++i) {
            for (int j = 0; j<NUM_DIMS; ++j) {
                fprintf(outfile, "%f ", gsl_matrix_get(means, i, j));
            }
            fprintf(outfile, "\n");
        }
        //print_matrix(means);

    }

    gsl_matrix_free(data);
    gsl_matrix_free(priors);
    gsl_matrix_free(means);
    gsl_matrix_free(posteriors);
    for (int i=0; i<NUM_COMPONENTS; ++i) {
        gsl_matrix_free(sigmas[i]);
    }


    rc = MPI_Finalize();
    return rc;

}


