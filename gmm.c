//
// Gaussian mixture model
//

#include <stdio.h>
#include <stdlib.h>
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
    double f;

    // read data
    unsigned int data_size = 0;

    // initial size of the array for the data points
    size_t data_array_size = 1024;
    gsl_vector** data_points = malloc(data_array_size * sizeof(*data_points));

    // read in the data
    FILE* infile = fopen("s1.txt", "r");
    if (infile == NULL) {
        perror("Error while opening the input data file.\n");
        exit(EXIT_FAILURE);
    }

    char *lineptr = NULL;
    char *startptr;
    char *endptr;
    size_t n = 0;
    i=0;
    // get every line in the input file
    while (getline(&lineptr, &n, infile) != -1) {
        data_points[i] = gsl_vector_alloc(DIM);
        ++data_size;

        // extract every double
        startptr = lineptr;
        for (j=0; j<DIM; ++j) {
            f = strtod(startptr, &endptr);
            if (startptr == endptr) {
                // no value was read, raise an error
                printf("Not enough components for input data point %d.\n", i+1);
                exit(EXIT_FAILURE);
            }
            startptr = endptr;

            // store the actual value
            gsl_vector_set(data_points[i], j, f);
        }
        // scan once more to issue a warning if there are more components
        // in the input data, than expected
        strtod(startptr, &endptr);
        if (startptr != endptr) {
            printf("Ignoring coordinate higher than DIM in input point %d.\n", i+1);
        }

        ++i;
        // increase the array size if necessary
        if (i >= data_array_size) {
            data_array_size += 1024;
            if ((data_points = reallocarray(data_points, data_array_size, sizeof(*data_points))) == NULL) {
                perror("Could not resize data_points array when reading input.\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    free(lineptr);
    lineptr = NULL;
    fclose(infile);

    //TODO: Check that NUM_COMPONENTS < data_size
    assert(NUM_COMPONENTS < data_size);
    //unsigned int DIM = 2;
    
    // reserve some space
    // "priors" a scalar for each component
    double priors[NUM_COMPONENTS];
    // posterior, a number for every pair of data point and component
    gsl_matrix* posterior = gsl_matrix_alloc(data_size, NUM_COMPONENTS);

    // "mus" the mean of each component
    gsl_vector* mus[NUM_COMPONENTS];
    for (i=0; i<NUM_COMPONENTS; ++i) {
        mus[i] = gsl_vector_alloc(DIM);
    }

    // finally reserve space for the sigmas, which are covariance matrices for each component
    gsl_matrix* sigmas[NUM_COMPONENTS];
    for (i=0; i<NUM_COMPONENTS; ++i) {
        sigmas[i] = gsl_matrix_alloc(DIM, DIM);
    }


    // initialise the means by "randomly" picking data points
    for (i=0; i<NUM_COMPONENTS; ++i) {
        gsl_vector_memcpy(mus[i], data_points[i]);
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

    // clean up temporary variables
    gsl_vector_free(tmp);
    gsl_vector_free(general_mean);
    gsl_matrix_free(general_sigma);

    // calculate
    for (i=0; i<50; ++i) {
        printf("step %d\n", i);
        // Expectation step
        expectation_step((const gsl_vector**) data_points, data_size, priors,
                            (const gsl_vector**) mus, (const gsl_matrix**) sigmas,
                            posterior);
        
        // Maximisation step
        maximization_step((const gsl_vector**) data_points, data_size, posterior,
                            priors, mus, sigmas);
    }


    // output to gmm_out.txt
    FILE* outfile = fopen("gmm_out.txt", "w");

    for (i = 0; i<NUM_COMPONENTS; ++i) {
        for (j = 0; j<DIM; ++j) {
            fprintf(outfile, "%f ", gsl_vector_get(mus[i], j));
        }
        fprintf(outfile, "\n");
    }
    fclose(outfile);

    // clean up
    for (i=0; i<data_size; ++i) {
        gsl_vector_free(data_points[i]);
    }
    free(data_points);
    for (i=0; i<NUM_COMPONENTS; ++i) {
        gsl_vector_free(mus[i]);
        gsl_matrix_free(sigmas[i]);
    }
    gsl_matrix_free(posterior);

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

    // clean up
    gsl_vector_free(workspace);
    gsl_vector_free(current_row);
}

void maximization_step(const gsl_vector** data, const unsigned int data_size,
                        const gsl_matrix* posterior,
                        double* new_priors, gsl_vector** new_means,
                        gsl_matrix** new_sigmas) {
    int i,k;
    // first calculate the component wise sum of the posterior
    double posterior_sums[NUM_COMPONENTS];
    for (k=0; k<NUM_COMPONENTS; ++k) {
        posterior_sums[k] = 0;
        for (i=0; i<data_size; ++i) {
            posterior_sums[k] += gsl_matrix_get(posterior, i, k);
        }
        new_priors[k] = posterior_sums[k] / (double) data_size;
    }

    // update the means
    gsl_vector* tmp = gsl_vector_alloc(DIM);
    for (k=0; k<NUM_COMPONENTS; ++k) {
        gsl_vector_set_zero(new_means[k]);
        for (i=0; i<data_size; ++i) {
            gsl_vector_memcpy(tmp, data[i]);
            gsl_vector_scale(tmp, gsl_matrix_get(posterior, i, k));
            gsl_vector_add(new_means[k], tmp);
        }
        gsl_vector_scale(new_means[k], 1/posterior_sums[k]);
    }

    // update the covariances
    for (k=0; k<NUM_COMPONENTS; ++k) {
        gsl_matrix_set_zero(new_sigmas[k]);
        for (i=0; i<data_size; ++i) {
            // translate the data according to current mean
            gsl_vector_memcpy(tmp, data[i]);
            gsl_vector_sub(tmp, new_means[k]);
            // tmp.T * tmp + general_sigma, stores the result in the lower triangle of general_sigma
            gsl_blas_dsyr(CblasLower, 1, tmp, new_sigmas[k]);
        }
        gsl_matrix_scale(new_sigmas[k], 1/posterior_sums[k]);
    }

    // clean up
    gsl_vector_free(tmp);
}
