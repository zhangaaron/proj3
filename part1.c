#include <emmintrin.h>
#include <nmmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel){

	/*
									  +----------------+
                                      | 00000000000000 |
                                      | 0 +---------+0 |
                                      | 0 |         |0 |
                                      | 0 |         |0 |
                                      | 0 |         |0 |
                                      | 0 |         |0 |
                                      | 0 +---------+0 |
                                      | 00000000000000 |
                                      +----------------+
	*/
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;


    //Determine padded matrix size: 
    int padded_matrix_size = (data_size_X + kern_cent_X) * (data_size_Y + kern_cent_Y);

    //Create a padded matrix: for simplicity's sake we're just iterating through the entire thing and zeroing everything. 
    //Optimize later?
    float padded_in[padded_matrix_size];
    __m128i zero_pad = _mm_setzero_si128(); //128 bit value with all zeros. 
    for (int i = 0; i < padded_matrix_size; i += 16) {
    	*(__m128i*)(padded_in + i + 0) = zero_pad;
    	*(__m128i*)(padded_in + i + 4) = zero_pad;
    	*(__m128i*)(padded_in + i + 8) = zero_pad;
    	*(__m128i*)(padded_in + i + 12) = zero_pad; 

    }
    //Pad tail
    for (int i = padded_matrix_size / 16 * 16; i < padded_matrix_size;  i ++) {
    	padded_in[i] = 0;
    }
	__m128 array_elems_to_load0;
	__m128 array_elems_to_load1;
	__m128 array_elems_to_load2;
	__m128 array_elems_to_load3;
    for (int i = 0; i < data_size_Y; i ++ ) {
    	for (int j = 0; j < data_size_X; j += 16 ) {
    		//get 4 128bit quantities per iteration of loop from in. 
    		array_elems_to_load0 = _mm_load_ps (in + i * data_size_X + j + 0);
    		array_elems_to_load1 = _mm_load_ps (in + i * data_size_X + j + 4);
    		array_elems_to_load2 = _mm_load_ps (in + i * data_size_X + j + 8);
    		array_elems_to_load3 = _mm_load_ps (in + i * data_size_X + j+ 12);
    		//Put array elements into padded array. 
    		*(__m128i*)(padded_in + (i + kern_cent_Y) * (data_size_X + 2 * kern_cent_X) + kern_cent_X + j + 0) = array_elems_to_load0;
    		*(__m128i*)(padded_in + (i + kern_cent_Y) * (data_size_X + 2 * kern_cent_X) + kern_cent_X + j + 4) = array_elems_to_load4;
    		*(__m128i*)(padded_in + (i + kern_cent_Y) * (data_size_X + 2 * kern_cent_X) + kern_cent_X + j + 8) = array_elems_to_load8;
    		*(__m128i*)(padded_in + (i + kern_cent_Y) * (data_size_X + 2 * kern_cent_X) + kern_cent_X + j + 12) = array_elems_to_load12;

    	}
    	//clean-up tail for stragglers
    	for (int j = data_size_X/16 * 16; j < data_size_X; j++) {
    		padded_in[(i + kern_cent_Y) * (data_size_X + 2 * kern_cent_X) + kern_cent_X + j] = in[i * data_size_X + j];
    	}
    }



    //Test
    // for (int i = 0; i < data_size_X; i ++) {
    // 	if (padded_in[i] != 0.0f) {
    // 		printf("Found elem not zero at 0x%x\n", i );
    // 	}
    // 	if (i % 10000 == 0) {
    // 		printf("Success!");
    // 	}
    // }

    
    // main convolution loop
	for(int x = 0; x < data_size_X; x++){ // the x coordinate of the output location we're focusing on
		for(int y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
			for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
				for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
					// only do the operation if not out of bounds
					if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
						//Note that the kernel is flipped
						out[x+y*data_size_X] += 
								kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X];
					}
				}
			}
		}
	}
	return 1;
}
