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
    int padded_matrix_size = data_size_X * data_size_Y + 2 * data_size_Y + 2 * data_size_X - 2;

    //Create a padded matrix:
    //Pad the top with zeros:
    float padded_in[padded_matrix_size];
    __m128i zero_pad = _mm_setzero_si128(); //128 bit value with all zeros. 
    for (int i = 0; i < data_size_X; i += 16) {
    	*(__m128i*)(padded_in + i + 0) = zero_pad;
    	*(__m128i*)(padded_in + i + 4) = zero_pad;
    	*(__m128i*)(padded_in + i + 8) = zero_pad;
    	*(__m128i*)(padded_in + i + 12) = zero_pad; 

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
