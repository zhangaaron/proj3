#include <emmintrin.h>
#include <nmmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
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

    int pad_X = (data_size_X + 2 * kern_cent_X);
    int pad_Y = (data_size_Y + 2 * kern_cent_Y);



    //Determine padded matrix size: 
    int padded_matrix_size = (pad_X) * (pad_Y);

    //Pad the top kern y + first row's kernx
    float *padded_in;
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

    //Vectorized-unrolled method of placing items into array and padding. 
    for (int j = 0; j < data_size_Y; j ++ ) {
    	// for (int i = 0; i < data_size_X; i += 16 ) {
    		            
     //        *(__m128*)(padded_in + i + kern_cent_X + (j + kern_cent_Y) * (data_size_X + 2 * kern_cent_X) + 0)  = _mm_loadu_ps (in + j * data_size_X + i + 0); 
     //        printf("HI!\n");     	
     //    }
    	//clean-up tail
    	for (int tail_counter= 0; tail_counter < data_size_X; tail_counter++) {
    		padded_in[tail_counter+ kern_cent_X + (j + kern_cent_Y) * (data_size_X + 2 * kern_cent_X)] = in[tail_counter + j * data_size_X];
    	}

        //Padded zeros at end and front of each row.
        for (int k = data_size_X; k < data_size_X + kern_cent_X * 2; k++ ) {
            padded_in[k + kern_cent_X + (j + kern_cent_X)] = 0;
        }
    }

    for (int i = (data_size_Y + kern_cent_Y) * (data_size_X + 2 * kern_cent_X); i < padded_matrix_size ; i += 16) {
        *(__m128*)(padded_in + i + 0) = zero_pad;
        *(__m128*)(padded_in + i + 4) = zero_pad;
        *(__m128*)(padded_in + i + 8) = zero_pad;
        *(__m128*)(padded_in + i + 12) = zero_pad;


    }
    //Pad tail
    for (int i = (data_size_Y + kern_cent_Y) * (data_size_X + 2 * kern_cent_X)/ 16 * 16; i < padded_matrix_size;  i ++) {
        padded_in[i] = 0;
    }


    free(padded_in);
    padded_in = NULL; 
    return 1;
}

int print_matrix(float* matrix_to_print, int data_size_X, int data_size_Y) {
	for (int j = 0; j < data_size_Y; j ++ ) { 
		for (int i = 0; i < data_size_X; i ++) {
		printf("%f ", matrix_to_print[i  + data_size_X * j]);
		}
		printf("/n");
	}
}

int main() {
	float small_array[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
	float output_array[9];
	print_matrix(small_array, 3, 3);
	conv2D(small_array, output_array, 3, 3, small_array);

}
