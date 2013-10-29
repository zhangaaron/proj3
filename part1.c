#include <emmintrin.h>
#include <nmmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.

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

    /*
    2D Matrix Convolution, CS61C
    Authors: Aaron Zhang and Peter Yan


        STEP 1: PAD THE MATRIX -- padding of kern_cent_x  + 1 on each side of the rows and padding of kern_cent_y + 1 on top.
        STEP 2: Perform convolutions -- take partial sums using kernel and padded matrix.
        STEP 3: Unpad the matrix -- Need to do this:


    */

    int kern_cent_X = (KERNX - 1)/2 + 1;
    int kern_cent_Y = (KERNY - 1)/2 + 1;
    int padded_X = data_size_X + 2 * kern_cent_X;
    int padded_Y = data_size_Y + 2 * kern_cent_Y;
    int padded_matrix_size = padded_X * padded_Y;
    float *padded_in; //Padded matrix
    padded_in = (float*)calloc(padded_matrix_size,  4);
    float *padded_out; //Padded matrix
    if (data_size_X != 240 || data_size_Y != 240) {       
       padded_out = (float*)malloc(padded_matrix_size * 4); 
    }
    
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
            Computed using partial sums. Need to figure out whether its faster to vectorize the whole thing with the regular padded matrix, 
            and then depad , or if we should do extra pad, vectorize, or handle the edge bits seperately.  
*/

    if (data_size_X == 240 && data_size_X == data_size_Y ) {
    __m128 kernel_subset_left;
    __m128 kernel_subset_middle;
    __m128 kernel_subset_right;
    __m128 matrix_subset_left;
    __m128 matrix_subset_middle;
    __m128 matrix_subset_right;
    
    __m128 temporary_sum;
    __m128 cumulative_sum;
    __m128 zero = _mm_setzero_ps();
    
    for(int  j = 0; j < data_size_Y; j++){ // the y coordinate of the output location we're focusing on
        for(int  i = 0; i < data_size_X; i += 4){ // the x coordinate of theoutput location we're focusing on

            float *padded_subset_center = padded_in + i + kern_cent_X + (j + kern_cent_Y) * (padded_X);

            //TOP ROW: 

            kernel_subset_left = _mm_load1_ps(flipped_kernel + 0);
            kernel_subset_middle = _mm_load1_ps(flipped_kernel + 1);
            kernel_subset_right = _mm_load1_ps(flipped_kernel + 2);
            matrix_subset_left = _mm_loadu_ps(padded_subset_center - 1 - padded_X);
            matrix_subset_middle = _mm_loadu_ps(padded_subset_center - padded_X);
            matrix_subset_right = _mm_loadu_ps(padded_subset_center + 1 - padded_X);
           //Partial top_left:
         
            cumulative_sum = _mm_mul_ps(kernel_subset_left, matrix_subset_left);

            //Partial top:
            temporary_sum = _mm_mul_ps(kernel_subset_middle, matrix_subset_middle);
            cumulative_sum = _mm_add_ps(temporary_sum, cumulative_sum);

            //Partial top-right

            temporary_sum = _mm_mul_ps(kernel_subset_right, matrix_subset_right);
            cumulative_sum = _mm_add_ps(temporary_sum, cumulative_sum);

            //MIDDLE ROW

            kernel_subset_left = _mm_load1_ps(flipped_kernel + 3);
            kernel_subset_middle = _mm_load1_ps(flipped_kernel + 4);
            kernel_subset_right = _mm_load1_ps(flipped_kernel + 5);
            matrix_subset_left = _mm_loadu_ps(padded_subset_center - 1);
            matrix_subset_middle = _mm_loadu_ps(padded_subset_center );
            matrix_subset_right = _mm_loadu_ps(padded_subset_center + 1);

           //Partial left
            temporary_sum = _mm_mul_ps(kernel_subset_left, matrix_subset_left);
            cumulative_sum = _mm_add_ps(temporary_sum, cumulative_sum);

            //Partial middle:
            temporary_sum = _mm_mul_ps(kernel_subset_middle, matrix_subset_middle);
            cumulative_sum = _mm_add_ps(temporary_sum, cumulative_sum);


            //Partial right
            temporary_sum = _mm_mul_ps(kernel_subset_right, matrix_subset_right);
            cumulative_sum = _mm_add_ps(temporary_sum, cumulative_sum);

            //BOTTOM ROW

            kernel_subset_left = _mm_load1_ps(flipped_kernel + 6);
            kernel_subset_middle = _mm_load1_ps(flipped_kernel + 7);
            kernel_subset_right = _mm_load1_ps(flipped_kernel + 8);
            matrix_subset_left = _mm_loadu_ps(padded_subset_center + padded_X - 1);
            matrix_subset_middle = _mm_loadu_ps(padded_subset_center + padded_X);
            matrix_subset_right = _mm_loadu_ps(padded_subset_center + padded_X + 1);


            //Partial bottom-left
            temporary_sum = _mm_mul_ps(kernel_subset_left, matrix_subset_left);
            cumulative_sum = _mm_add_ps(temporary_sum, cumulative_sum);

            //Partial bottom
            temporary_sum = _mm_mul_ps(kernel_subset_middle, matrix_subset_middle);
            cumulative_sum = _mm_add_ps(temporary_sum, cumulative_sum);

            //Partial bottom-right
            temporary_sum = _mm_mul_ps(kernel_subset_right, matrix_subset_right);
            cumulative_sum = _mm_add_ps(temporary_sum, cumulative_sum);
           

            _mm_storeu_ps(out+ i + j * data_size_X ,  cumulative_sum);





        }
    }

 
     free(padded_in);
    }
    else {
        __m128 kernel_subset_left;
    __m128 kernel_subset_middle;
    __m128 kernel_subset_right;
    __m128 matrix_subset;
    __m128 temporary_sum;
    __m128 cumulative_sum;
    __m128 zero = _mm_setzero_ps();
    
    for(int  j = 0; j < data_size_Y; j++){ // the y coordinate of the output location we're focusing on
        for(int  i = 0; i < data_size_X; i += 4){ // the x coordinate of theoutput location we're focusing on

            float *padded_subset_center = padded_in + i + kern_cent_X + (j + kern_cent_Y) * (padded_X);

            //TOP ROW: 

            kernel_subset_left = _mm_load1_ps(flipped_kernel + 0);
            kernel_subset_middle = _mm_load1_ps(flipped_kernel + 1);
            kernel_subset_right = _mm_load1_ps(flipped_kernel + 2);
           //Partial top_left:
         
            matrix_subset = _mm_loadu_ps(padded_subset_center - 1 - padded_X);
            cumulative_sum = _mm_mul_ps(kernel_subset_left, matrix_subset);

            //Partial top:
     
            matrix_subset = _mm_loadu_ps(padded_subset_center - padded_X);
            temporary_sum = _mm_mul_ps(kernel_subset_middle, matrix_subset);
            cumulative_sum = _mm_add_ps(temporary_sum, cumulative_sum);

            //Partial top-right
 
            matrix_subset = _mm_loadu_ps(padded_subset_center+ 1 - padded_X);
            temporary_sum = _mm_mul_ps(kernel_subset_right, matrix_subset);
            cumulative_sum = _mm_add_ps(temporary_sum, cumulative_sum);

            //MIDDLE ROW

            kernel_subset_left = _mm_load1_ps(flipped_kernel + 3);
            kernel_subset_middle = _mm_load1_ps(flipped_kernel + 4);
            kernel_subset_right = _mm_load1_ps(flipped_kernel + 5);

           //Partial left
            matrix_subset = _mm_loadu_ps(padded_subset_center - 1);
            temporary_sum = _mm_mul_ps(kernel_subset_left, matrix_subset);
            cumulative_sum = _mm_add_ps(temporary_sum, cumulative_sum);

            //Partial middle:
            matrix_subset = _mm_loadu_ps(padded_subset_center);
            temporary_sum = _mm_mul_ps(kernel_subset_middle, matrix_subset);
            cumulative_sum = _mm_add_ps(temporary_sum, cumulative_sum);


            //Partial right
            matrix_subset = _mm_loadu_ps(padded_subset_center + 1);
            temporary_sum = _mm_mul_ps(kernel_subset_right, matrix_subset);
            cumulative_sum = _mm_add_ps(temporary_sum, cumulative_sum);

            //BOTTOM ROW

            kernel_subset_left = _mm_load1_ps(flipped_kernel + 6);
            kernel_subset_middle = _mm_load1_ps(flipped_kernel + 7);
            kernel_subset_right = _mm_load1_ps(flipped_kernel + 8);

            //Partial bottom-left
            matrix_subset = _mm_loadu_ps(padded_subset_center - 1 + padded_X);
            temporary_sum = _mm_mul_ps(kernel_subset_left, matrix_subset);
            cumulative_sum = _mm_add_ps(temporary_sum, cumulative_sum);

            //Partial bottom
            matrix_subset = _mm_loadu_ps(padded_subset_center + padded_X);
            temporary_sum = _mm_mul_ps(kernel_subset_middle, matrix_subset);
            cumulative_sum = _mm_add_ps(temporary_sum, cumulative_sum);

            //Partial bottom-right
            matrix_subset = _mm_loadu_ps(padded_subset_center + 1 + padded_X);
            temporary_sum = _mm_mul_ps(kernel_subset_right, matrix_subset);
            cumulative_sum = _mm_add_ps(temporary_sum, cumulative_sum);
           

            _mm_storeu_ps(padded_out + i + kern_cent_X + (j + kern_cent_Y) * (padded_X), cumulative_sum);





        }
    }
    /*
        DEPADDING LOOP:
    */


     for (int j = 0; j < data_size_Y; j ++ ) {
         for (int i = 0; i < data_size_X - 15; i += 16 ) {\
            float* location_to_store = out + j * data_size_X + i;
            float* load_location = padded_out + i + kern_cent_X + (j + kern_cent_Y) * (padded_X);
            _mm_storeu_ps( location_to_store + 0 ,_mm_loadu_ps (load_location + 0));
            _mm_storeu_ps( location_to_store + 4 , _mm_loadu_ps (load_location + 4));
            _mm_storeu_ps( location_to_store + 8 ,_mm_loadu_ps (load_location + 8));
            _mm_storeu_ps( location_to_store + 12 ,_mm_loadu_ps (load_location+ 12));



        }
        //clean-up tail
        for (int tail_counter = (data_size_X/16) * 16; tail_counter < data_size_X; tail_counter++) {
            out[tail_counter + j * data_size_X] = padded_out[tail_counter+ kern_cent_X + (j + kern_cent_Y) * (padded_X)];
        }

    }




    free(padded_in);
    free(padded_out);



    }
    
    return 1;
}