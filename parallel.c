//#include "gmmnew.c"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include <mpi.h>

const unsigned int NUM_COMPONENTS;
unsigned int NUM_DIMS;
unsigned int NUM_DATA;

// our rank and the toal number of processors as global variables
int rank, num_processors;

size_t min(size_t a, size_t b) {
    if (a < b) {
        return a;
    }
    else {
        return b;
    }
}

double parallel_sum(const int component_i, const gsl_matrix* data,
gsl_matrix* means, const gsl_matrix* posteriors, gsl_matrix* new_sigma) {

    // new_sigma and posterior_sum will be the output of the sum,
    // we need to collect these via recursive doubling after the parallel computation
    double posterior_sum = 0;
    gsl_matrix* local_sigma = gsl_matrix_calloc(NUM_DIMS, NUM_DIMS);

    // first compute our range of the data that we need to sum over
    size_t local_size = NUM_DATA / num_processors;
    if (rank < NUM_DATA % num_processors) {
        ++local_size;
    }
    size_t local_offset = rank * NUM_DATA / num_processors + min(rank, NUM_DATA % num_processors);

    gsl_matrix* xx = gsl_matrix_alloc(NUM_DIMS,NUM_DIMS);
    gsl_matrix* x_cpy = gsl_matrix_alloc(1,NUM_DIMS);
    for (int data_i = local_offset; data_i < local_offset + local_size; data_i++)
    {
        gsl_matrix_const_view x_view = gsl_matrix_const_submatrix(data,data_i,0,1,NUM_DIMS);
        gsl_matrix_const_view u_view = gsl_matrix_const_submatrix(means,component_i,0,1,NUM_DIMS);
        gsl_matrix_memcpy(x_cpy,&x_view.matrix);
        gsl_matrix_sub(x_cpy,&u_view.matrix);

        gsl_blas_dgemm(CblasTrans,CblasNoTrans, gsl_matrix_get(posteriors,data_i,component_i) ,x_cpy,x_cpy,0,xx);
        //gsl_matrix_scale(xx,gsl_matrix_get(posteriors,data_i,component_i));

        gsl_matrix_add(local_sigma,xx);
        posterior_sum += gsl_matrix_get(posteriors,data_i,component_i);

    }

    // now do the recursive doubling process
    // we need to send the posterior_sum as well as new_sigma
    double result = 0;
    MPI_Allreduce(&posterior_sum, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    int i,j;
    for (i=0; i < NUM_DIMS; ++i) {
        for (j=0; j<NUM_DIMS; ++j) {
            MPI_Allreduce(gsl_matrix_ptr(local_sigma, i, j), gsl_matrix_ptr(new_sigma, i, j),
            1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
    }


    // free allocated space
    gsl_matrix_free(xx);
    gsl_matrix_free(x_cpy);


    return result;
}
