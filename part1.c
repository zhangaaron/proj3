#include <emmintrin.h>
#include <nmmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.

int print_matrix(float* matrix_to_print, int data_size_X, int data_size_Y) {
    printf("Here's your matrix!\n");
    for (int j = 0; j < data_size_Y; j ++ ) { 
        for (int i = 0; i < data_size_X; i ++) {
            if (matrix_to_print[i  + data_size_X * j] >= 0.0) {
                printf(" ");
            }
        printf("%0.1f ", matrix_to_print[i  + data_size_X * j]);
        }
        printf("\n");
    }
    printf("\n");
}

int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel){

    /*
    2D Matrix Convolution, CS61C
    Authors: Aaron Zhang and Peter Yan


        STEP 1: PAD THE MATRIX -- padding of kern_cent_x on each side of the rows and padding of kern_cent_y on top.
        STEP 2: Perform convolutions -- take partial sums using kernel and padded matrix.
        STEP 3: Unpad the matrix -- DONT ACTUALLY NEED TO


    */

    int kern_cent_X = (KERNX - 1)/2;
    int kern_cent_Y = (KERNY - 1)/2;
    int padded_X = data_size_X + 2 * kern_cent_X;
    int padded_Y = data_size_Y + 2 * kern_cent_Y;
    int padded_matrix_size = padded_X * padded_Y;
    float *padded_in; //Padded matrix
    padded_in = (float*)calloc(padded_matrix_size,  4);
    //Vectorized-unrolled method of placing items into array and padding.
    for (int j = 0; j < data_size_Y; j ++ ) {
         for (int i = 0; i < data_size_X - 15; i += 16 ) {
            _mm_storeu_ps((padded_in + i + kern_cent_X + (j + kern_cent_Y) * (padded_X) + 0),_mm_loadu_ps (in + j * data_size_X + i + 0));
            _mm_storeu_ps((padded_in + i + kern_cent_X + (j + kern_cent_Y) * (padded_X) + 4),_mm_loadu_ps (in + j * data_size_X + i + 4));
            _mm_storeu_ps((padded_in + i + kern_cent_X + (j + kern_cent_Y) * (padded_X) + 8),_mm_loadu_ps (in + j * data_size_X + i + 8));
            _mm_storeu_ps((padded_in + i + kern_cent_X + (j + kern_cent_Y) * (padded_X) + 12),_mm_loadu_ps (in + j * data_size_X + i + 12));

        }
        //clean-up tail
        for (int tail_counter = (data_size_X/16) * 16; tail_counter < data_size_X; tail_counter++) {
            //printf("Putting in data at %d in padded matrix from %d\n", tail_counter+ kern_cent_X + (j + kern_cent_Y) * (data_size_X + 2 * kern_cent_X), tail_counter + j * data_size_X);
            padded_in[tail_counter+ kern_cent_X + (j + kern_cent_Y) * (padded_X)] = in[tail_counter + j * data_size_X];
        }

    }
    //Flip kernel
    float flipped_kernel[KERNX * KERNY];
    for (int i = 0; i < KERNX * KERNY; i++) {
        flipped_kernel[(KERNX * KERNY - 1) - i] = kernel[i];
    }
/*



   +-------------------------------------------------------------------------------------------------------------------------------------------------+
            +---------------------+
            |MAIN CONVOLUTION LOOP|
            +---------------------+
*/
    for(int y = 0; y < data_size_Y; y++){ // the x coordinate of the output location we're focusing on
        for(int x = 0; x < data_size_X; x++){ // the y coordinate of theoutput location we're focusing on
            for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped x coordinate
                for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped y coordinate
                    // padding means we never go out of bounds!
                   
                    out[x+y*data_size_X] += 
                                kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * padded_in[x + i + kern_cent_X + (y + j + kern_cent_Y) * (data_size_X + 2 * kern_cent_X)];

                }
            }
        }
    }
    //print_matrix(out, data_size_X, data_size_Y);





    free(padded_in); 
    return 1;
}