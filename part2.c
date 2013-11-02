#include <emmintrin.h>
#include <nmmintrin.h>
#include <omp.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.


    /*
    2D Matrix Convolution, CS61C
    Authors: Aaron Zhang and Peter Yan
    
    A highly optimized method of image processing using data and thread level parallelism.

    Featuring SSE intrinsics and OpenMP. 


    */

int print_matrix(float* matrix_to_print, int data_size_X, int data_size_Y) {
    printf("\n");
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

int print_vector(__m128 a) {
    float *printout = (float*)(&a);
    printf("Vector is %0.1f %0.1f %0.1f %0.1f\n", printout[0], printout[1], printout[2], printout[3]);
}

int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel){

    omp_set_num_threads(16);
    int kern_cent_X = (KERNX - 1)/2 + 1;
    int kern_cent_Y = (KERNY - 1)/2 + 1;


    int padded_X = data_size_X + 2 * kern_cent_X;
    int padded_Y = data_size_Y + 2 * kern_cent_Y;
    int padded_matrix_size = padded_X * padded_Y;
    float padded_in[padded_matrix_size]; //Padded matrix


    //Make a padded matrix.
    #pragma omp parallel
    {

    #pragma omp for
        for (int i = 0; i < padded_matrix_size - 3; i += 4) {
            _mm_storeu_ps(padded_in + i, _mm_setzero_ps());        
        }

        for (int i = padded_matrix_size/ 4 * 4; i < padded_matrix_size; i++) {
            padded_in[i] = 0;
        }
    #pragma omp for schedule(dynamic)
        for (int j = 0; j < data_size_Y; j ++ ) {
         for (int i = 0; i < data_size_X - 15; i += 16 ) {
            _mm_storeu_ps((padded_in + i + kern_cent_X + (j + kern_cent_Y) * (padded_X) + 0),_mm_loadu_ps (in + j * data_size_X + i + 0));
            _mm_storeu_ps((padded_in + i + kern_cent_X + (j + kern_cent_Y) * (padded_X) + 4),_mm_loadu_ps (in + j * data_size_X + i + 4));
            _mm_storeu_ps((padded_in + i + kern_cent_X + (j + kern_cent_Y) * (padded_X) + 8),_mm_loadu_ps (in + j * data_size_X + i + 8));
            _mm_storeu_ps((padded_in + i + kern_cent_X + (j + kern_cent_Y) * (padded_X) + 12),_mm_loadu_ps (in + j * data_size_X + i + 12));

            }
        //clean-up tail
        for (int tail_counter = (data_size_X/16) * 16; tail_counter < data_size_X; tail_counter++) {
            padded_in[tail_counter+ kern_cent_X + (j + kern_cent_Y) * (padded_X)] = in[tail_counter + j * data_size_X];
            }

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
            Computed using partial sums. Does the computation for one cell of the kernel for four adjacent array elements at the same time using SSE 
            intrinsics. Multiple writes to a _m128 register cumulative sum, but only one write to the main array at the end of the loop. Reads from the 
            matrix 9 times. Reads kernel cells which are stored in _m128 values 9 times. 


   +--------------------------------------------------------------------------------------------------------------------------------------------------+
*/

    
    __m128 kernel_subset_topleft = _mm_load1_ps(flipped_kernel + 0);
    __m128 kernel_subset_topmiddle = _mm_load1_ps(flipped_kernel + 1);
    __m128 kernel_subset_topright = _mm_load1_ps(flipped_kernel + 2);

    __m128 kernel_subset_left = _mm_load1_ps(flipped_kernel + 3);
    __m128 kernel_subset_middle = _mm_load1_ps(flipped_kernel + 4);
    __m128 kernel_subset_right = _mm_load1_ps(flipped_kernel + 5);

    __m128 kernel_subset_botleft = _mm_load1_ps(flipped_kernel + 6);
    __m128 kernel_subset_botmiddle = _mm_load1_ps(flipped_kernel + 7);
    __m128 kernel_subset_botright = _mm_load1_ps(flipped_kernel + 8);

    #pragma omp parallel
    {
    __m128 cumulative_sum;
    __m128 matrix_subset_left;
    __m128 matrix_subset_middle;
    __m128 matrix_subset_right;    
    #pragma omp for private(cumulative_sum, matrix_subset_right, matrix_subset_left, matrix_subset_middle)
    for(int  j = 0; j < data_size_Y - 1; j++){ // the y coordinate of the output location we're focusing on        
        for(int  i = 0; i < data_size_X; i += 4){ // the x coordinate of theoutput location we're focusing on
           
            float *padded_subset_center = padded_in + i + kern_cent_X + (j + kern_cent_Y) * (padded_X);

            //TOP ROW: 
            matrix_subset_left = _mm_loadu_ps(padded_subset_center - 1 - padded_X);
            matrix_subset_middle = _mm_loadu_ps(padded_subset_center - padded_X);
            matrix_subset_right = _mm_loadu_ps(padded_subset_center + 1 - padded_X);
           //Partial top_left:         
            cumulative_sum = _mm_mul_ps(kernel_subset_topleft, matrix_subset_left);
            //Partial top:
            cumulative_sum = _mm_add_ps(_mm_mul_ps(kernel_subset_topmiddle, matrix_subset_middle), cumulative_sum);
            //Partial top-right
            cumulative_sum = _mm_add_ps(_mm_mul_ps(kernel_subset_topright, matrix_subset_right), cumulative_sum);


            //MIDDLE ROW
            matrix_subset_left = _mm_loadu_ps(padded_subset_center - 1);
            matrix_subset_middle = _mm_loadu_ps(padded_subset_center);
            matrix_subset_right = _mm_loadu_ps(padded_subset_center + 1);
            //Partial left
            cumulative_sum = _mm_add_ps(_mm_mul_ps(kernel_subset_left, matrix_subset_left), cumulative_sum);
            //Partial middle:
            cumulative_sum = _mm_add_ps(_mm_mul_ps(kernel_subset_middle, matrix_subset_middle), cumulative_sum);
            //Partial right 
            cumulative_sum = _mm_add_ps(_mm_mul_ps(kernel_subset_right, matrix_subset_right), cumulative_sum);
            
            //BOTTOM ROW
            matrix_subset_left = _mm_loadu_ps(padded_subset_center + padded_X - 1);
            matrix_subset_middle = _mm_loadu_ps(padded_subset_center + padded_X);
            matrix_subset_right = _mm_loadu_ps(padded_subset_center + padded_X + 1);
            //Partial bottom-left
            cumulative_sum = _mm_add_ps(_mm_mul_ps(kernel_subset_botleft, matrix_subset_left), cumulative_sum);
            //Partial bottom
            cumulative_sum = _mm_add_ps(_mm_mul_ps(kernel_subset_botmiddle, matrix_subset_middle), cumulative_sum);
            //Partial bottom-right
            cumulative_sum = _mm_add_ps(_mm_mul_ps(kernel_subset_botright, matrix_subset_right), cumulative_sum);           

            _mm_storeu_ps(out+ i + j * data_size_X ,  cumulative_sum);

        }
       
    }
    #pragma omp for
    for(int  i = 0; i < data_size_X - 3; i += 4){ // the x coordinate of theoutput location we're focusing on
            __m128 cumulative_sum;
            __m128 matrix_subset_left;
            __m128 matrix_subset_middle;
            __m128 matrix_subset_right;
            float *padded_subset_center = padded_in + i + kern_cent_X + (data_size_Y - 1 + kern_cent_Y) * (padded_X);

            //TOP ROW: 
            matrix_subset_left = _mm_loadu_ps(padded_subset_center - 1 - padded_X);
            matrix_subset_middle = _mm_loadu_ps(padded_subset_center - padded_X);
            matrix_subset_right = _mm_loadu_ps(padded_subset_center + 1 - padded_X);
           //Partial top_left:         
            cumulative_sum = _mm_mul_ps(kernel_subset_topleft, matrix_subset_left);
            //Partial top:
            cumulative_sum = _mm_add_ps(_mm_mul_ps(kernel_subset_topmiddle, matrix_subset_middle), cumulative_sum);
            //Partial top-right
            cumulative_sum = _mm_add_ps(_mm_mul_ps(kernel_subset_topright, matrix_subset_right), cumulative_sum);


            //MIDDLE ROW
            matrix_subset_left = _mm_loadu_ps(padded_subset_center - 1);
            matrix_subset_middle = _mm_loadu_ps(padded_subset_center);
            matrix_subset_right = _mm_loadu_ps(padded_subset_center + 1);
            //Partial left
            cumulative_sum = _mm_add_ps(_mm_mul_ps(kernel_subset_left, matrix_subset_left), cumulative_sum);
            //Partial middle:
            cumulative_sum = _mm_add_ps(_mm_mul_ps(kernel_subset_middle, matrix_subset_middle), cumulative_sum);
            //Partial right 
            cumulative_sum = _mm_add_ps(_mm_mul_ps(kernel_subset_right, matrix_subset_right), cumulative_sum);
            
            //BOTTOM ROW
            matrix_subset_left = _mm_loadu_ps(padded_subset_center + padded_X - 1);
            matrix_subset_middle = _mm_loadu_ps(padded_subset_center + padded_X);
            matrix_subset_right = _mm_loadu_ps(padded_subset_center + padded_X + 1);
            //Partial bottom-left
            cumulative_sum = _mm_add_ps(_mm_mul_ps(kernel_subset_botleft, matrix_subset_left), cumulative_sum);
            //Partial bottom
            cumulative_sum = _mm_add_ps(_mm_mul_ps(kernel_subset_botmiddle, matrix_subset_middle), cumulative_sum);
            //Partial bottom-right
            cumulative_sum = _mm_add_ps(_mm_mul_ps(kernel_subset_botright, matrix_subset_right), cumulative_sum);           

            _mm_storeu_ps(out+ i + (data_size_Y - 1) * data_size_X ,  cumulative_sum);

        }
}
     //partial sums tail.
        float c_sum;
        for (int i = data_size_X/ 4 * 4; i < data_size_X; i ++) {
            int padded_subset_center = i + kern_cent_X + (data_size_Y  - 1 + kern_cent_Y) * (padded_X);
            //top partials
            c_sum = padded_in[padded_subset_center - 1 - padded_X] * flipped_kernel[0];  
            c_sum += padded_in[padded_subset_center - padded_X] * flipped_kernel[1];
            c_sum += padded_in[padded_subset_center + 1 - padded_X] * flipped_kernel[2];
            //middle partials
            c_sum += padded_in[padded_subset_center -1 ] * flipped_kernel[3];
            c_sum += padded_in[padded_subset_center ] * flipped_kernel[4];
            c_sum += padded_in[padded_subset_center + 1] * flipped_kernel[5];
            //bottom partials
            c_sum += padded_in[padded_subset_center - 1 + padded_X] * flipped_kernel[6];
            c_sum += padded_in[padded_subset_center + padded_X] * flipped_kernel[7];
            c_sum += padded_in[padded_subset_center + 1 + padded_X] * flipped_kernel[8];
            out[i + (data_size_Y - 1) * data_size_X] = c_sum;
        }

 
    
    return 1;
}