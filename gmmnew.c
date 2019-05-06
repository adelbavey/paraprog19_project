//
// Gaussian mixture model
//

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_randist.h>

// global constants, feel free to configure
//
// how many components to try to fit onto the data
const unsigned int NUM_COMPONENTS = 4;
unsigned int NUM_DIMS = 0;
unsigned int NUM_DATA = 0;


int get_N_D(void){
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    size_t read;

    fp = fopen("s2.txt", "r");
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


void initialize_components(const gsl_matrix* data, 
gsl_matrix* priors, gsl_matrix* means, gsl_matrix** sigmas){

    //Initialize priors
    //gsl_vector * priors = gsl_vector_alloc(NUM_COMPONENTS);
    gsl_matrix_set_all(priors, 1.0/NUM_COMPONENTS);

    //Init means (non random)
    //gsl_matrix * means = gsl_matrix_alloc(NUM_COMPONENTS,NUM_DIMS);
    for (int i = 0; i < NUM_COMPONENTS; i++)
    {   gsl_vector tmp = gsl_matrix_const_row(data,i).vector;
        gsl_matrix_set_row(means,i,&tmp);
    }

    //Init covariance matrices.
    //Create initital sigma shared by all components
    //gsl_matrix* sigmas[NUM_COMPONENTS];

    gsl_matrix* init_sigma = gsl_matrix_alloc(NUM_DIMS,NUM_DIMS);
    for (int i = 0; i < NUM_DIMS; i++)
    {
        for (int j = 0; j < NUM_DIMS; j++)
        {
            double cov = gsl_stats_covariance(gsl_matrix_const_column(data,i).vector.data,1,gsl_matrix_const_column(data,j).vector.data,1,NUM_DATA);
            gsl_matrix_set(init_sigma,i,j,cov);
        }
    }
    
    for (int i = 0; i < NUM_COMPONENTS; i++)
    {
        sigmas[i] = gsl_matrix_alloc(NUM_DIMS,NUM_DIMS);
        gsl_matrix_memcpy(sigmas[i],init_sigma);
    }


}

void expectation_step(const gsl_matrix* data, 
const gsl_matrix* priors, const gsl_matrix* means, gsl_matrix** sigmas,
gsl_matrix* posteriors){

    gsl_vector * work = gsl_vector_alloc(NUM_DIMS);
    double result = 0;

    for (int data_i = 0; data_i < NUM_DATA; data_i++)
    {
        for (int component_i = 0; component_i < NUM_COMPONENTS; component_i++)
        {
            gsl_vector_const_view x_view = (gsl_matrix_const_row(data,data_i));
            gsl_vector_const_view mean_view = (gsl_matrix_const_row(data,component_i));
            const gsl_matrix* sigma = sigmas[component_i];

            //Cholesky
            gsl_matrix * sigma_chol = gsl_matrix_alloc(NUM_DIMS,NUM_DIMS);
            gsl_matrix_memcpy(sigma_chol,sigma);
            //print_matrix(sigma);
            gsl_linalg_cholesky_decomp(sigma_chol);

            //Get pdf value
            gsl_ran_multivariate_gaussian_pdf(&x_view.vector,&mean_view.vector,sigma_chol,&result,work);

            gsl_matrix_set(posteriors,data_i,component_i,gsl_matrix_get(priors,0,component_i)*result);
        }

        //Normalize the row in posteriors
        gsl_vector posterior_row = (gsl_matrix_row(posteriors,data_i).vector);
        double row_sum = 0;
        gsl_vector* ones = gsl_vector_alloc(NUM_COMPONENTS);
        gsl_vector_set_all(ones, 1);
        gsl_blas_ddot(&posterior_row, ones,&row_sum);
        gsl_vector_scale(&posterior_row,1.0/row_sum);
        gsl_matrix_set_row(posteriors,data_i,&posterior_row);
        
    }
    

}

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

        for (int data_i = 0; data_i < NUM_DATA; data_i++)
        {
            gsl_matrix_const_view x_view = gsl_matrix_const_submatrix(data,data_i,0,1,NUM_DIMS);
            gsl_matrix_const_view u_view = gsl_matrix_const_submatrix(means,component_i,0,1,NUM_DIMS);
            gsl_matrix* x_cpy = gsl_matrix_alloc(1,NUM_DIMS);
            gsl_matrix_memcpy(x_cpy,&x_view.matrix);
            gsl_matrix_sub(x_cpy,&u_view.matrix);
            gsl_matrix* xx = gsl_matrix_alloc(NUM_DIMS,NUM_DIMS);
            gsl_blas_dgemm(CblasTrans,CblasNoTrans,1,x_cpy,x_cpy,0,xx);
            gsl_matrix_scale(xx,gsl_matrix_get(posteriors,data_i,component_i));

            gsl_matrix_add(new_sigmas[component_i],xx);
            posterior_sum += gsl_matrix_get(posteriors,data_i,component_i);
        }

        gsl_matrix_scale(new_sigmas[component_i],1.0/posterior_sum);
        
        gsl_matrix_memcpy(sigmas[component_i],new_sigmas[component_i]);
    }

    
}

int main(int argl, char* argv[]){

    //Get number of data points and dims
    get_N_D();

    //Get data
    FILE * fp = fopen("s2.txt", "r");
    gsl_matrix * data = gsl_matrix_alloc (NUM_DATA, NUM_DIMS);
    gsl_matrix_fscanf(fp, data);

    //Allocate component memory and initialize
    gsl_matrix * priors = gsl_matrix_alloc(1,NUM_COMPONENTS);
    gsl_matrix * means = gsl_matrix_alloc(NUM_COMPONENTS,NUM_DIMS);
    gsl_matrix* sigmas[NUM_COMPONENTS];
    gsl_matrix* posteriors = gsl_matrix_alloc(NUM_DATA,NUM_COMPONENTS);
    initialize_components(data,priors,means,sigmas);
    

    //EM ALGORITHM
    for (int iter = 0; iter < 5; iter++)
    {
        print_matrix(posteriors);
        printf("\n");
        expectation_step(data,priors,means,sigmas,posteriors);
        
        maximization_step(data,priors,means,sigmas,posteriors);
        
    }
    //print_matrix(means);
    


}


