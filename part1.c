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
        STEP 3: Unpad the matrix -- take some scissors and cut the original matrix out of the padded matrix.


    */
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;
    int padded_X = data_size_X + 2 * kern_cent_X;
    int padded_Y = data_size_Y + 2 * kern_cent_Y;

    //print_matrix(in, data_size_X, data_size_Y);
    //Determine padded matrix size: 
    int padded_matrix_size = padded_X * padded_Y;

    //padded_in and padded_out are the padded versions of the in and out matricies.
    float *padded_in;
    float *padded_out;
    padded_out = (float*) malloc(padded_matrix_size * 4);
    padded_in = (float*)malloc(padded_matrix_size * 4);
    __m128 zero_pad = _mm_setzero_ps();  //128 bit value with all zeros. 
    for (int i = 0; i < kern_cent_Y * (data_size_X + 2 * kern_cent_X) + kern_cent_X; i += 16) {
        *(__m128*)(padded_in + i + 0) = zero_pad;
        *(__m128*)(padded_in + i + 4) = zero_pad;
        *(__m128*)(padded_in + i + 8) = zero_pad;
        *(__m128*)(padded_in + i + 12) = zero_pad; 

    }
    //Pad tail

    for (int i = (kern_cent_Y * (data_size_X + 2 * kern_cent_X) + kern_cent_X)/ 16 * 16; i < kern_cent_Y * (data_size_X + 2 * kern_cent_X) + kern_cent_X;  i ++) {
        padded_in[i] = 0;
    }
    //printf("Padded matrix size is %d\n", padded_matrix_size );
    //Vectorized-unrolled method of placing items into array and padding. 
    for (int j = 0; j < data_size_Y; j ++ ) {
        // for (int i = 0; i < data_size_X; i += 16 ) {
                        
     //        *(__m128*)(padded_in + i + kern_cent_X + (j + kern_cent_Y) * (data_size_X + 2 * kern_cent_X) + 0)  = _mm_loadu_ps (in + j * data_size_X + i + 0); 
     //        printf("HI!\n");         
     //    }
        //clean-up tail
        for (int tail_counter= 0; tail_counter < data_size_X; tail_counter++) {
            //printf("Putting in data at %d in padded matrix from %d\n", tail_counter+ kern_cent_X + (j + kern_cent_Y) * (data_size_X + 2 * kern_cent_X), tail_counter + j * data_size_X);
            padded_in[tail_counter+ kern_cent_X + (j + kern_cent_Y) * (data_size_X + 2 * kern_cent_X)] = in[tail_counter + j * data_size_X];
        }
        //print_matrix(padded_in, padded_X, padded_Y);

        //Padded zeros at end and front of each row.
        for (int k = (padded_X) * (j + kern_cent_Y + 1) - 2 * kern_cent_X + 1; k < (padded_X) * (j + kern_cent_Y + 1); k++ ) {
            padded_in[k] = 0;
        }
        //print_matrix(padded_in, padded_X, padded_Y);
    }
    for (int vector_zero_pad_counter = (data_size_Y + kern_cent_Y) * padded_X; vector_zero_pad_counter < padded_matrix_size - 16; vector_zero_pad_counter += 16) {
        //printf("Putting in zeros at %d in padded matrix\n", vector_zero_pad_counter);
        *(__m128*)(padded_in + vector_zero_pad_counter + 0) = zero_pad;
        *(__m128*)(padded_in + vector_zero_pad_counter + 4) = zero_pad;
        *(__m128*)(padded_in + vector_zero_pad_counter + 8) = zero_pad;
        *(__m128*)(padded_in + vector_zero_pad_counter + 12) = zero_pad;


    }
    //The complicated formula here gets how much tail needs to be done.
    for (int i = padded_matrix_size -( 1 + padded_matrix_size - (data_size_Y + kern_cent_Y) * padded_X )% 16 ; i < padded_matrix_size;  i ++) {
            //printf("Tail is at %d\n", i );
            padded_in[i] = 0;
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