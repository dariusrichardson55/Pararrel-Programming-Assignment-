#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;
/* Short Summary

Host code
Step 1: I have assigned bins sizes to 256 for 


Kernels




*/

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : select image" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";
	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		
		//a 3x3 convolution mask implementing an averaging filter
		std::vector<float> convolution_mask = { 1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9 };

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input,"input");
		int image_size = image_input.size();
		int image_total = image_size*sizeof(unsigned char);




	/*	typedef int mytype;
		size_t local_size = 10;*/

		// number of bins (Histogram)
		vector<int> H(256);
		vector<int> CH(256);
		vector<int> NL(256);		

		// number of bins (cumulative histogram)
		

		// The size of the histogram is set to 256
		int Histogram_size = 256;

		// The histogram_total will be the size of the histogram times the size of int
		int Histogram_total = Histogram_size * sizeof(int);
		int cumulative_histogram_total = Histogram_size * sizeof(int);
		int normalise_total = Histogram_size * sizeof(int);
		int output_total = Histogram_size * sizeof(unsigned char);

		//The device buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_total);
		cl::Buffer histogram(context, CL_MEM_READ_ONLY, Histogram_total);
		cl::Buffer cumulative_histogram(context, CL_MEM_READ_ONLY, cumulative_histogram_total);
		cl::Buffer normalise(context, CL_MEM_READ_ONLY, normalise_total);
	    cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_total);

		/// A kernel operate each element of a stream and writes  to an output string 
		// The buffer for the input image to provide the total length of the image 
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_total, &image_input[0]);

		// (First step) In the buffer for the histogram to provide the total length of the histogrm

		queue.enqueueFillBuffer(histogram, 0, 0, Histogram_total);
		queue.enqueueFillBuffer(cumulative_histogram, 0, 0, cumulative_histogram_total);
		queue.enqueueFillBuffer(normalise, 0, 0, normalise_total);
		queue.enqueueFillBuffer(dev_image_output, 0, 0, output_total);

	    // In the buffer it displays the result for the length of the length on image  
		queue.enqueueFillBuffer(histogram, 0, 0, Histogram_total);

		
		
		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel = cl::Kernel(program, "the_hist_simple");	
		cl::Kernel kernel2 = cl::Kernel(program, "scan_add_atomic");
		cl::Kernel kernel3 = cl::Kernel(program, "normalise");
		cl::Kernel kernel4 = cl::Kernel(program, "reprojection");


		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, histogram);
		kernel2.setArg(0, histogram);
		kernel2.setArg(1, cumulative_histogram);
		kernel3.setArg(0, cumulative_histogram);
		kernel3.setArg(1, normalise);

		kernel4.setArg(0, dev_image_input);
		kernel4.setArg(1, dev_image_output);
		kernel4.setArg(2, normalise);

	

		CImg<unsigned char> output_image(image_filename.c_str());

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(CH.size()), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel3, cl::NullRange, cl::NDRange(NL.size()), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel4, cl::NullRange, cl::NDRange(output_image.size()), cl::NullRange);

		queue.enqueueReadBuffer(histogram, CL_TRUE, 0, Histogram_total, &H[0]);
		queue.enqueueReadBuffer(cumulative_histogram, CL_TRUE, 0, cumulative_histogram_total, &CH[0]);
		queue.enqueueReadBuffer(normalise, CL_TRUE, 0, normalise_total, &NL[0]);
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, image_total, &output_image[0]);
		



		cout << "Histogram" << endl;
		cout << H << endl;

		cout << "Cumulative Histogram" << endl;
		cout << CH << endl;

		cout << "Normalisation";
		cout << NL;

		CImgDisplay disp_output(output_image, "output");

		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}

	}

	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
