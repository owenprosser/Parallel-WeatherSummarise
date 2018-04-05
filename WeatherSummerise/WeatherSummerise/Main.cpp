#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

/* Vector library to enable vector use*/
#include <vector>
/* Libraries to read in data from files, needed as I'll be loading in the temperatures from a .txt file and kernel from a .cl file */
#include <iostream>
#include <fstream>

/* Load in string library (kind of an essential when dealing with strings)*/
#include <string>

/* Use ctime to deal with the time so it can be stored correctly in one vector reducing the amount of memory used */
#include <ctime>

/*OpenCL Library (Only need windows as this will only be running on windows)*/
#include <CL/cl.hpp>



using namespace std;

/*vector<string> get_data_from_line(string& line, string& delim) {
int index;
vector<string> data;
while ((index = line.find(delim)) != string::npos) {
cout << line << endl;
data.push_back(line.substr(0, index));
line.erase(0, index + delim.length());
}
return data;
}*/

vector<string> get_data_from_line(string line, char delim) {
	vector<string> data;
	string data_string;

	int last_delim = 0;

	for (int i = 0; i < line.size(); i++)
	{
		//Get every string before the delimiter and append it to the data vector 
		if (line[i] == delim) {
			data.push_back(data_string);
			//Once the data has been appended reset the temporary string variable so the next string can be added to it
			data_string = "";
			//Keep track of the index of the last delimiter found so the last element in the line can be appended at the end of the search
			last_delim = i;
		}
		//Append the character to data_string
		data_string += line[i];
	}
	//Get the last element in the line and add it to the data vector
	data.push_back(line.substr(last_delim));

	return data;
}







const char *getErrorString(cl_int error) {
	switch (error) {
		// run-time and JIT compiler errors
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
	}
}








int main(int argc, char **argv) {
	/*GET PLATFORMS*/
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		cout << "No platforms found!" << endl;
		exit(1);
	}
	cl::Platform default_platform;

	int num_platforms = platforms.size();

	cout << "Found: " << num_platforms << " platforms" << endl;

	cout << "Select a platform from the list below" << endl;

	for (int i = 0; i < num_platforms; i++)
	{
		cout << i << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << endl;
	}

	int selected_platform_number;

	cin >> selected_platform_number;

	default_platform = platforms[selected_platform_number];
	cout << "Using: " << default_platform.getInfo<CL_PLATFORM_NAME>() << endl;

	/* -------------------------------------------------- */

	/* GET DEVICES*/
	vector<cl::Device> devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if (devices.size() == 0) {
		cerr << "Can't find any devices" << endl;
		exit(1);
	}

	int num_devices = devices.size();

	cout << "Found: " << num_devices << " devices" << endl;

	cout << "Select a device from the list below" << endl;

	for (int i = 0; i < num_devices; i++)
	{
		cout << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << endl;
	}

	int selected_device_number;

	cin >> selected_device_number;

	cl::Device default_device = devices[selected_device_number];
	cout << "Using: " << default_device.getInfo<CL_DEVICE_NAME>() << endl;

	/* -------------------------------------------------- */

	/*LOAD IN DATA*/
	ifstream data_file;
	string line;
	string delim = " ";

	vector<vector<string>> file_data;

	/* Create vectors to hold the file data*/

	vector<int> station_name;
	vector<float> temperatures;
	vector<struct tm> date_time;
	//vector<float> temperatures = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };

	data_file.open("temp_lincolnshire_short.txt");

	if (!data_file) {
		cerr << "can't open the file";
		exit(1);
	}
	else {

		/* Run through every line and put the data in the right vectors, avoids using a seperate loop to save time*/
		while (getline(data_file, line)) {
			//Get the data from the line, split so it can be placed into the correct vectors
			vector<string> line_data = get_data_from_line(line, ' ');

			//cout << line_data[0] << endl;

			//cout << line_data.size() << ": " << line_data[line_data.size()-1] << endl;

			//Create the date_time structure and place it into the date_time vector
			struct tm date_time_struct;
			date_time_struct.tm_year = stoi(line_data[1]) - 1900; //Year is the year from 1900 so 1900 has to be taken from the year given in the file
			date_time_struct.tm_mon = stoi(line_data[2]) - 1; //Month must be >= 0 and <= 11 so 1 must be taken from the given month value
			date_time_struct.tm_mday = stoi(line_data[3]);
			date_time_struct.tm_wday = 0;
			date_time_struct.tm_yday = 0;
			date_time_struct.tm_hour = stoi(line_data[4].substr(0, 2)); //The hour is the first two numbers in the time section of the file
			date_time_struct.tm_min = stoi(line_data[4].substr(2, 2)); //The minute is the two numbers following the hour in the time section
			date_time_struct.tm_sec = 0; //No second value is given so assumed to be 0

			date_time.push_back(date_time_struct);

			//Get the temperature and put it into the temperature structure
			temperatures.push_back(stof(line_data[5]));
		}
		data_file.close();
	}

	/* -------------------------------------------------- */

	/* Run the kernel functions to do the data processing */

	try {
		//GET CONTEXT FROM DEFAULT DEVICE
		cl::Context context{ { default_device } };


		//CREATE COMMAND QUEUE
		cl::CommandQueue command_queue(context);

		//Add sources
		ifstream kernelFile("kernals.cl");
		string kernelAsString(istreambuf_iterator<char>(kernelFile), (istreambuf_iterator<char>()));

		cl::Program::Sources programSource(1, make_pair(kernelAsString.c_str(), kernelAsString.length() + 1));

		cl::Program program(context, programSource);

		//Build imported kernel to devices
		try {
			program.build(devices);
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}


		cl::Kernel addKernel(program, "addAll");

		size_t sizeOfInput = temperatures.size() * sizeof(float);
		size_t local_size = 10;
		size_t padding_size = temperatures.size() % local_size;


		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<float> A_ext(local_size - padding_size, 0.0);
			//append that extra vector to our input
			temperatures.insert(temperatures.end(), A_ext.begin(), A_ext.end());
		}


		//Create output vector
		vector<float> output(temperatures.size());
		size_t sizeOfOutput = output.size() * sizeof(float);

		size_t number_of_groups = temperatures.size() / local_size;

		//Create buffers
		cl::Buffer temperatures_buffer(context, CL_MEM_READ_WRITE, sizeOfInput);
		//cl::Buffer B_buffer(context, CL_MEM_READ_WRITE, sizeOfB);
		cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, sizeOfOutput);
		//Create queue and copy vectors to device memory
		cl::CommandQueue queue(context);

		queue.enqueueWriteBuffer(temperatures_buffer, CL_TRUE, 0, sizeOfInput, &temperatures[0]);
		queue.enqueueFillBuffer(output_buffer, 0, 0, sizeOfOutput);

		//Execute kernel
		addKernel.setArg(0, temperatures_buffer);
		addKernel.setArg(1, output_buffer);
		addKernel.setArg(2, cl::Local(local_size * sizeof(float)));
		//addKernel.setArg(3, cl::Local(local_size * sizeof(float)));


		queue.enqueueNDRangeKernel(addKernel, cl::NullRange, cl::NDRange(temperatures.size()), cl::NDRange(local_size));

		queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeOfOutput, &output[0]);

		//float sum_p = (float)output[0]/1000.0;
		float sum_p = output[0];

		/*for(int i = 0; i < temperatures.size(); i++)
		{
		printf("\nInput: %f\n", temperatures[i]);
		printf("Output: %f\n", output[i]);
		}*/

		float mean = sum_p / temperatures.size();

		cout << "Sum: " << sum_p << endl;

		cout << "Mean: " << mean << endl;

	}
	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}

	/* PROGRAM EXIT */
	cout << "\n\nType 'q' and press enter to exit" << endl;

	char wait_for_q;
	cin >> wait_for_q;

	while (wait_for_q != 'q') {
		//do nothing
	}

	return 0;
}