//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}


kernel void filter_r(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	B[id] = A[id];
}
// The histogram 
kernel void the_hist_simple(global const unsigned char* A, global int* H) {
	int id = get_global_id(0); 	//assumes that H has been initialised to 0
	int counter = 0;
	//int bin_index = A[id];//take value as a bin index
	
	for (int i=0; i < get_global_size(0); i++){
		if (A[i] == id) {
			counter = counter + 1;
		}	
	}
	H[id] = counter;
}

//The Cumulative Histogram
// atomic add for scan 
kernel void scan_add_atomic(global int* A, global int* CH) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id + 1; i < N; i++)
		atomic_add(&CH[i], A[id]);
}

kernel void normalise(global int* CH, global int* NH) {

int id = get_global_id(0);
float x = CH[id];
int max = CH[255];
int min = CH[0];

float equation = (x-min)/(max-min); 

  NH[id] = equation * 255;

}


kernel void reprojection(global const uchar* input, global uchar* output, global const int* LUT) {

  int id = get_global_id(0); 
  int x = input[id];
  int x_reprojection = LUT[x];

  output[id] = x_reprojection;

}



//simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	B[id] = A[id];
}

//2D averaging filter
kernel void avg_filter2D(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	ushort result = 0;

	for (int i = (x-1); i <= (x+1); i++)
	for (int j = (y-1); j <= (y+1); j++) 
		result += A[i + j*width + c*image_size];

	result /= 9;

	B[id] = (uchar)result;
}

//2D 3x3 convolution kernel
kernel void convolution2D(global const uchar* A, global uchar* B, constant float* mask) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	ushort result = 0;

	for (int i = (x-1); i <= (x+1); i++)
	for (int j = (y-1); j <= (y+1); j++) 
		result += A[i + j*width + c*image_size]*mask[i-(x-1) + j-(y-1)];

	B[id] = (uchar)result;
}