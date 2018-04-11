#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <CL/cl.hpp>

using namespace std;

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

static void average() {
	ifstream data_file;
	string line;
	string delim = " ";

	vector<vector<string>> file_data;

	/* Vectors used to hold the data read in from the file*/

	vector<int> temp;
	vector<struct tm> date_time;

	/*GET PLATFORMS*/
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		cout << "Error. No openCL devices" << endl;
		exit(1);
	}
	cl::Platform default_platform;

	int number_of_platforms = platforms.size();

	cout << number_of_platforms << " platforms available on this machine:" << endl;

	cout << "Select Platform:" << endl;

	for (int i = 0; i < number_of_platforms; i++)
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

	cout << "Select a device: " << endl;

	for (int i = 0; i < num_devices; i++)
	{
		cout << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << endl;
	}

	int selected_device_number;

	cin >> selected_device_number;

	cl::Device default_device = devices[selected_device_number];
	cout << "Using: " << default_device.getInfo<CL_DEVICE_NAME>() << endl;


	data_file.open("temp_lincolnshire.txt");

	if (!data_file) { //if the specified txt file can't be found
		cerr << "can't open the file";
		exit(1);
	}
	else {

		/* Process each line of the txt file to get temperature values*/
		while (getline(data_file, line)) {
			//Gets each of the lines of data and splits them at the  ' ' so can be incremented
			vector<string> line_data = get_data_from_line(line, ' ');

			//get the temperature values from the 5th element of vector
			float __temp = stof(line_data[5]);
			//push back temperature values to vector
			temp.push_back(int(__temp*100));
		}
		data_file.close();
	}

	cout << date_time.size() << "\n";

	cl::Context context{ { default_device } };

	//CREATE COMMAND QUEUE
	cl::CommandQueue command_queue(context);

	//Add the .cl containing all kernels
	ifstream kernelFile("OpenCLFile.cl");
	string kernelAsString(istreambuf_iterator<char>(kernelFile), (istreambuf_iterator<char>()));

	cl::Program::Sources programSource(1, make_pair(kernelAsString.c_str(), kernelAsString.length() + 1));

	cl::Program program(context, programSource);

	program.build(devices);

	cl::Kernel reduce_add(program, "reduce_add_4");

	size_t sizeOfInput = temp.size() * sizeof(float);
	size_t local_size = 10;
	size_t padding_size = temp.size() % local_size;


	if (padding_size) {
		//create vector of 0 values
		std::vector<float> A_ext(local_size - padding_size, 0.0);
		//add this vector to the end of the temperature vector
		temp.insert(temp.end(), A_ext.begin(), A_ext.end());
	}

	//Create output vector
	vector<int> output(temp.size());
	size_t sizeOfOutput = output.size() * sizeof(int);

	//size_t number_of_groups = temp.size() / local_size;

	//Create buffers
	cl::Buffer temperatures_buffer(context, CL_MEM_READ_WRITE, sizeOfInput);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, sizeOfOutput);

	//Create queue and copy vectors to device memory
	cl::CommandQueue queue(context);

	queue.enqueueWriteBuffer(temperatures_buffer, CL_TRUE, 0, sizeOfInput, &temp[0]);
	queue.enqueueFillBuffer(output_buffer, 0, 0, sizeOfOutput);

	//Execute kernel
	reduce_add.setArg(0, temperatures_buffer);
	reduce_add.setArg(1, output_buffer);
	reduce_add.setArg(2, cl::Local(local_size * sizeof(float)));

	queue.enqueueNDRangeKernel(reduce_add, cl::NullRange, cl::NDRange(temp.size()), cl::NDRange(local_size));

	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeOfOutput, &output[0]);

	float sum_p = output[0];

	printf("%i total temperature records \n", temp.size());

	float mean = (sum_p / temp.size()) / 100;

	cout << "Sum: " << sum_p << endl;

	cout << "Mean: " << mean << endl;
}

static void minMax() {
	ifstream data_file;
	string line;
	string delim = " ";

	vector<vector<string>> file_data;

	/* Create vectors to hold the file data*/

	vector<int> location;
	vector<int> temp;
	vector<struct tm> date_time;

	
	/*GET PLATFORMS*/
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		cout << "Error. No openCL devices" << endl;
		exit(1);
	}
	cl::Platform default_platform;

	int number_of_platforms = platforms.size();

	cout << number_of_platforms << " platforms available on this machine:" << endl;

	cout << "Select Platform:" << endl;

	for (int i = 0; i < number_of_platforms; i++)
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

	cout << "Select a device: " << endl;

	for (int i = 0; i < num_devices; i++)
	{
		cout << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << endl;
	}

	int selected_device_number;

	cin >> selected_device_number;

	cl::Device default_device = devices[selected_device_number];
	cout << "Using: " << default_device.getInfo<CL_DEVICE_NAME>() << endl;


	data_file.open("temp_lincolnshire.txt");

	if (!data_file) {
		cerr << "can't open the file";
		exit(1);
	}
	else {

		/* Run through every line and put the data in the right vectors, avoids using a seperate loop to save time*/
		while (getline(data_file, line)) {
			//Get the data from the line, split so it can be placed into the correct vectors
			vector<string> line_data = get_data_from_line(line, ' ');

			//Get the temperature and put it into the temperature structure
			temp.push_back(stof(line_data[5]));
		}
		data_file.close();
	}

	cout << date_time.size() << "\n";

	cl::Context context{ { default_device } };

	//CREATE COMMAND QUEUE
	cl::CommandQueue command_queue(context);

	//Add sources
	ifstream kernelFile("OpenCLFile.cl");
	string kernelAsString(istreambuf_iterator<char>(kernelFile), (istreambuf_iterator<char>()));

	cl::Program::Sources programSource(1, make_pair(kernelAsString.c_str(), kernelAsString.length() + 1));

	cl::Program program(context, programSource);

	//Build imported kernel to devices
	program.build(devices);

	cl::Kernel reduce_min(program, "reduce_min");
	cl::Kernel reduce_max(program, "reduce_max");

	size_t sizeOfInput = temp.size() * sizeof(float);
	size_t local_size = 10;
	size_t padding_size = temp.size() % local_size;


	if (padding_size) {
		//create an extra vector with neutral values
		std::vector<float> A_ext(local_size - padding_size, 0.0);
		//append that extra vector to our input
		temp.insert(temp.end(), A_ext.begin(), A_ext.end());
	}

	//Create output vector
	vector<int> output(temp.size());
	size_t sizeOfOutput = output.size() * sizeof(float);

	//size_t number_of_groups = temp.size() / local_size;

	//Create buffers
	cl::Buffer temperatures_buffer(context, CL_MEM_READ_WRITE, sizeOfInput);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, sizeOfOutput);

	//Create queue and copy vectors to device memory
	cl::CommandQueue queue(context);

	queue.enqueueWriteBuffer(temperatures_buffer, CL_TRUE, 0, sizeOfInput, &temp[0]);
	queue.enqueueFillBuffer(output_buffer, 0, 0, sizeOfOutput);

	//Execute kernel
	reduce_min.setArg(0, temperatures_buffer);
	reduce_min.setArg(1, output_buffer);
	reduce_min.setArg(2, cl::Local(local_size * sizeof(float)));

	queue.enqueueNDRangeKernel(reduce_min, cl::NullRange, cl::NDRange(temp.size()), cl::NDRange(local_size));

	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeOfOutput, &output[0]);

	float minimum = output[0];

	cout << "Minimum Temperature: " << minimum << endl;

	reduce_max.setArg(0, temperatures_buffer);
	reduce_max.setArg(1, output_buffer);
	reduce_max.setArg(2, cl::Local(local_size * sizeof(float)));

	queue.enqueueNDRangeKernel(reduce_max, cl::NullRange, cl::NDRange(temp.size()), cl::NDRange(local_size));

	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeOfOutput, &output[0]);

	float maximum = output[0];


	cout << "Maximum Temperature: " << maximum << endl;

	//cout << "Mean: " << mean << endl;

}

static void histogram() {
	ifstream data_file;
	string line;
	string delim = " ";

	vector<vector<string>> file_data;

	/* Create vectors to hold the file data*/

	vector<int> location;
	vector<int> temp;
	vector<struct tm> date_time;

	/*GET PLATFORMS*/
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		cout << "Error. No openCL devices" << endl;
		exit(1);
	}
	cl::Platform default_platform;

	int number_of_platforms = platforms.size();

	cout << number_of_platforms << " platforms available on this machine:" << endl;

	cout << "Select Platform:" << endl;

	for (int i = 0; i < number_of_platforms; i++)
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

	cout << "Select a device: " << endl;

	for (int i = 0; i < num_devices; i++)
	{
		cout << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << endl;
	}

	int selected_device_number;

	cin >> selected_device_number;

	cl::Device default_device = devices[selected_device_number];
	cout << "Using: " << default_device.getInfo<CL_DEVICE_NAME>() << endl;


	data_file.open("temp_lincolnshire.txt");

	if (!data_file) {
		cerr << "can't open the file";
		exit(1);
	}
	else {

		/* Run through every line and put the data in the right vectors, avoids using a seperate loop to save time*/
		while (getline(data_file, line)) {
			//Get the data from the line, split so it can be placed into the correct vectors
			vector<string> line_data = get_data_from_line(line, ' ');

			float __temp = stof(line_data[5]);
			//Get the temperature and put it into the temperature structure
			temp.push_back(int(__temp * 100));
		}
		data_file.close();
	}

	cout << date_time.size() << "\n";

	cl::Context context{ { default_device } };

	//CREATE COMMAND QUEUE
	cl::CommandQueue command_queue(context);

	//Add sources
	ifstream kernelFile("OpenCLFile.cl");
	string kernelAsString(istreambuf_iterator<char>(kernelFile), (istreambuf_iterator<char>()));

	cl::Program::Sources programSource(1, make_pair(kernelAsString.c_str(), kernelAsString.length() + 1));

	cl::Program program(context, programSource);

	//Build imported kernel to devices
	program.build(devices);

	cl::Kernel hist_simple(program, "hist");

	size_t sizeOfInput = temp.size() * sizeof(float);
	size_t local_size = 10;
	size_t padding_size = temp.size() % local_size;


	if (padding_size) {
		//create an extra vector with neutral values
		std::vector<float> A_ext(local_size - padding_size, 0.0);
		//append that extra vector to our input
		temp.insert(temp.end(), A_ext.begin(), A_ext.end());
	}

	//Create output vector
	vector<int> output(temp.size());
	size_t sizeOfOutput = output.size() * sizeof(float);

	size_t number_of_groups = temp.size() / local_size;

	float minimum = -25;
	float maximum = 45;

	int bin_no = (maximum - minimum);
	int binSizes = (maximum - minimum) / bin_no;

	vector<int> historgram_vector(bin_no);

	//Create buffers
	cl::Buffer temperatures_buffer(context, CL_MEM_READ_WRITE, sizeOfInput);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, sizeOfOutput);

	//Create queue and copy vectors to device memory
	cl::CommandQueue queue(context);

	queue.enqueueWriteBuffer(temperatures_buffer, CL_TRUE, 0, sizeOfInput, &temp[0]);
	queue.enqueueFillBuffer(output_buffer, 0, 0, sizeOfOutput);

	printf("executing Kernel\n");

	//Execute kernel
	hist_simple.setArg(0, temperatures_buffer);
	printf("1 \n");
	hist_simple.setArg(1, output_buffer);
	printf("2 \n");
	hist_simple.setArg(2, sizeof(int), &binSizes);
	printf("3 \n");
	hist_simple.setArg(3, sizeof(float), &minimum);
	printf("4 \n");
	hist_simple.setArg(4, sizeof(float), &maximum);
	printf("5 \n");

	queue.enqueueNDRangeKernel(hist_simple, cl::NullRange, cl::NDRange(temp.size()), cl::NDRange(local_size));

	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeOfOutput, &historgram_vector[0]);


	for each (int bin in output) {
		printf("%d \n", output);
	}

}

int main(int argc, char **argv) {

	while (true) {
		cout << "1. Average. \n";
		cout << "2. Minimum and Maximum Temperatures. \n";
		cout << "3. Temperatures Histogram. \n";
		cout << "q. Quit. \n";

		char choice;
		cin >> choice;

		if (choice == '1') {
			average();
		}
		else if (choice == '2') {
			minMax();
		}
		else if (choice == '3') {
			histogram();
		}
		else if (choice == 'q') {
			exit(0);
		}
	}
}