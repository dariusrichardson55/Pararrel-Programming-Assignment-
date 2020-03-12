#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

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

		// number of bins (cumulative histogram)
		vector<int> CH(256);

		// The size of the histogram is set to 256
		int Histogram_size = 256;
		// The histogram_total will be the size of the histogram times the size of int
		int Histogram_total = Histogram_size * sizeof(int);
		int Cumulative_histogram_total = Histogram_total * sizeof(H);
		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_total);
		cl::Buffer histogram(context, CL_MEM_READ_ONLY, Histogram_total);
		cl::Buffer Cumulative_histogram(context, CL_MEM_READ_ONLY, Cumulative_histogram_total);
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, Histogram_total);

		/// A kernel operate each element of a stream and writes the to an output string 
		// The buffer for the input image to provide the total length of the image 
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, Histogram_total, &H[0]);
		queue.enqueueWriteBuffer(histogram, CL_TRUE, 0, Cumulative_histogram_total, &CH[0]);
		// (First step) In the buffer for the histogram to provide the total length of the histogrm
		queue.enqueueFillBuffer(histogram, 0, 0, Histogram_total);
		queue.enqueueFillBuffer(Cumulative_histogram, 0, 0, Cumulative_histogram_total);
		// (Second step) In the buffer for the cuumaltiative histogram 
	//	queue.enqueueFillBuffer(cumulative_histogram, 0, 0, Cumulative_histogram_total);

 
	    // In the buffer it displays the result for the length of the length on image  
		queue.enqueueFillBuffer(dev_image_output, 0, 0, image_total);
		queue.enqueueFillBuffer(histogram, 0, 0, Cumulative_histogram_total);

	

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel = cl::Kernel(program, "the_hist_simple");
	

		
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, histogram);
		kernel.setArg(2, Cumulative_histogram);

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(CH.size()), cl::NullRange);

		queue.enqueueReadBuffer(histogram, CL_TRUE, 0, Histogram_total, &CH[0]);
		cout << H;

	//	queue.enqueueReadBuffer(cumulative_histogram, CL_TRUE, 0, Cumulative_histogram_total, &CH[0]);
		cout << CH;
	
		/*
		vector<unsigned char> output_buffer(image_input.size());
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image,"output");
		   
 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }		
		*/
		
	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
