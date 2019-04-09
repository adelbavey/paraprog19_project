//
// Gaussian mixture model
//

#include <stdio.h>
#include <assert.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_blas.h>


// global constants, feel free to configure
//
// how many components to try to fit onto the data
const unsigned int NUM_COMPONENTS = 4;
// how many dimensions do the data points have
const unsigned int DIM = 2;

// expectation step
// requires the data, priors, means and sigmas
void expectation_step();

// maximisation step
// requires the data and posteriors from the expecation step
void maximization_step();

int main(int argl, char* argv[]) {
    int i,j;

    // read data
    // for now use this static one
    gsl_vector* data_points[5];
    for (i=0; i<5; ++i) {
        data_points[i] = gsl_vector_alloc(2);
    }
    gsl_vector_set(data_points[0], 0, 602013);
    gsl_vector_set(data_points[0], 1, 574722);
    gsl_vector_set(data_points[1], 0, 627968);
    gsl_vector_set(data_points[1], 1, 574625);
    gsl_vector_set(data_points[2], 0, 607269);
    gsl_vector_set(data_points[2], 1, 536961);
    gsl_vector_set(data_points[3], 0, 603145);
    gsl_vector_set(data_points[3], 1, 574795);
    gsl_vector_set(data_points[4], 0, 671919);
    gsl_vector_set(data_points[4], 1, 571761);

    unsigned int data_size = 5;
    //TODO: Check that NUM_COMPONENTS < data_size
    assert(NUM_COMPONENTS < data_size);
    //unsigned int DIM = 2;
    
    // reserve some space
    // "priors" a scalar for each component
    double priors[NUM_COMPONENTS];

    // "mu" the mean of each component
    gsl_vector* mu[NUM_COMPONENTS];
    for (i=0; i<NUM_COMPONENTS; ++i) {
        mu[i] = gsl_vector_alloc(DIM);
    }

    // finally reserve space for the sigmas, which are covariance matrices for each component
    gsl_matrix* sigma[NUM_COMPONENTS];
    for (i=0; i<NUM_COMPONENTS; ++i) {
        sigma[i] = gsl_matrix_alloc(DIM, DIM);
    }


    // initialise the means by "randomly" picking data points
    for (i=0; i<NUM_COMPONENTS; ++i) {
        for (j=0; j<DIM; ++j) {
            mu[i][j] = data_points[i][j];
        }
    }

    // compute an initial variance-covariance which is the same for every component
    gsl_vector* general_mean = gsl_vector_calloc(DIM);
    for (i=0; i<data_size; ++i) {
        gsl_vector_add(general_mean, data_points[i]);
    }
    // create a temporary vector to use as divisor
    gsl_vector* tmp = gsl_vector_alloc(DIM);
    for (i=0; i<DIM; ++i) {
        gsl_vector_set_all(tmp, data_size);
    }
    // each element is now sum_i/N
    gsl_vector_div(general_mean, tmp);

    // now we've got the mean and can calculate the covariance matrix
    gsl_matrix* general_sigma = gsl_matrix_calloc(DIM, DIM);
    for (i=0; i<data_size; ++i) {
        gsl_vector_memcpy(tmp, data_points[i]);
        gsl_vector_sub(tmp, general_mean);
        // tmp.T * tmp + general_sigma, stores the result in the lower triangle of general_sigma
        gsl_blas_dsyr(CblasLower, 1, tmp, general_sigma);
    }
    gsl_matrix_scale(general_sigma, 1/(double)data_size);
    gsl_vector_free(tmp);

    // calculate
    //TODO:


    // clean up
    for (i=0; i<data_size; ++i) {
        gsl_vector_free(data_points[i]);
    }
    for (i=0; i<NUM_COMPONENTS; ++i) {
        gsl_vector_free(mu[i]);
        gsl_matrix_free(sigma[i]);
    }

    return 0;
}
