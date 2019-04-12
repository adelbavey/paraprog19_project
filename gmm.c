//
// Gaussian mixture model
//

#include <stdio.h>
#include <assert.h>

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
// how many dimensions do the data points have
const unsigned int DIM = 2;

/*
 * expectation step
 * requires the data, priors, means and sigmas
 * data is expected to be an array of vectors of size data_size
 * priors is expected to be an array of doubles of size NUM_COMPONENTS
 * means is expected to be an array of gsl_vectors of size NUM_COMPONENTS
 * sigma is expected to be an array of gsl_matrices of size NUM_COMPONENTS
 *      the matrices are expected to be given by cholesky decomposition, so
 *      only the lower triangle of sigma[k] will be used
 * posterior is expected to be a matrix of dimensions (data_size, NUM_COMPONENTS)
 *      and will be written to
 */
void expectation_step(const gsl_vector** data, const unsigned int data_size,
                        const double* priors, const gsl_vector** means,
                        const gsl_matrix** sigmas, gsl_matrix* posterior);

/*
 * maximisation step
 * requires the data and posterior from the expectation step
 * data is expected to be an array of vectors of size data_size
 * posterior is expected to be a matrix of dimensions (data_size, NUM_COMPONENTS)
 * 
 * the following parameters will be written to
 * new_priors is expected to be an array of doubles of size NUM_COMPONENTS
 * new_means is expected to be an array of vectors of size NUM_COMPONENTS
 * new_sigmas is expected to be an array of matrices of size NUM_COMPONENTS
 *      the matrices are expected to have dimensions (data_size, NUM_COMPONENT)
 *      the new covariance matrices will be given by their cholesky decomposition
 *      in the lower triangular, the strict upper triangle is undefined
 */
void maximization_step(const gsl_vector** data, const unsigned int data_size,
                        const gsl_matrix* posterior,
                        double* new_priors, gsl_vector** new_means,
                        gsl_matrix** new_sigmas);

int main(int argl, char* argv[]) {
    int i,j;

    // read data
    // for now use this static one
    gsl_vector* data_points[5];
    for (i=0; i<5; ++i) {
        data_points[i] = gsl_vector_alloc(DIM);
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
    // posterior, a number for every pair of data point and component
    gsl_matrix* posterior = gsl_matrix_alloc(data_size, NUM_COMPONENTS);

    // "mu" the mean of each component
    gsl_vector* mu[NUM_COMPONENTS];
    for (i=0; i<NUM_COMPONENTS; ++i) {
        mu[i] = gsl_vector_alloc(DIM);
    }

    // finally reserve space for the sigmas, which are covariance matrices for each component
    gsl_matrix* sigmas[NUM_COMPONENTS];
    for (i=0; i<NUM_COMPONENTS; ++i) {
        sigmas[i] = gsl_matrix_alloc(DIM, DIM);
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
    // each element is now sum_i/N
    gsl_vector_scale(general_mean, 1/(double) data_size);

    // create a temporary vector to use as divisor
    gsl_vector* tmp = gsl_vector_alloc(DIM);
    // now we've got the mean and can calculate the covariance matrix
    gsl_matrix* general_sigma = gsl_matrix_calloc(DIM, DIM);
    for (i=0; i<data_size; ++i) {
        gsl_vector_memcpy(tmp, data_points[i]);
        gsl_vector_sub(tmp, general_mean);
        // tmp.T * tmp + general_sigma, stores the result in the lower triangle of general_sigma
        gsl_blas_dsyr(CblasLower, 1, tmp, general_sigma);
    }
    gsl_matrix_scale(general_sigma, 1/(double)data_size);

    // the gsl function to calculate the pdf requires the covariance matrix
    // given in their cholesky decomposition, the lower triangle of the
    // output from gsl_linalg_cholesky_decomp1();
    //TODO: check error, maybe matrix not positive definite?
    gsl_linalg_cholesky_decomp1(general_sigma);
    // finally initialise the array of sigmas with this initial covariance
    for (i=0; i<NUM_COMPONENTS; ++i) {
        gsl_matrix_memcpy(sigmas[i], general_sigma);
    }
    gsl_vector_free(tmp);

    // calculate
	// Expectation step
    expectation_step((const gsl_vector**) data_points, data_size, priors,
                        (const gsl_vector**) mu, (const gsl_matrix**) sigmas,
                        posterior);
    
    // Maximisation step
	//priors, mu, Sigma = maximization_step(data, posterior)


    // clean up
    for (i=0; i<data_size; ++i) {
        gsl_vector_free(data_points[i]);
    }
    for (i=0; i<NUM_COMPONENTS; ++i) {
        gsl_vector_free(mu[i]);
        gsl_matrix_free(sigmas[i]);
    }

    return 0;
}


void expectation_step(const gsl_vector** data, const unsigned int data_size,
                        const double* priors, const gsl_vector** means,
                        const gsl_matrix** sigmas, gsl_matrix* posterior) {
    int i,k;
    double probability;
    gsl_vector* workspace = gsl_vector_alloc(DIM);
    gsl_vector* current_row = gsl_vector_alloc(NUM_COMPONENTS);
    for (i=0; i<data_size; ++i) {
        double row_sum = 0;
        for (k=0; k<NUM_COMPONENTS; ++k) {
            gsl_ran_multivariate_gaussian_pdf(data[i], means[k], sigmas[k], &probability, workspace);
            gsl_vector_set(current_row, k, priors[k]*probability);
            row_sum += priors[k]*probability;
        }
        gsl_vector_scale(current_row, 1/row_sum);
        gsl_matrix_set_row(posterior, i, current_row);
    }
}
