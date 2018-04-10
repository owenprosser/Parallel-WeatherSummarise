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

	/* Create vectors to hold the file data*/

	vector<int> location;
	vector<float> temp;
	vector<struct tm> date_time;

	/*GET PLATFORMS*/
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		cout << "Error. No openCL devices" << endl;
		exit(1);
	}
	cl::Platform default_platform;

	int num_platforms = platforms.size();

	cout << num_platforms << " platforms available on this machine:" << endl;

	cout << "Select Platform:" << endl;

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

	cout << "Found: " << num_devices << " device(s)" << " on " << default_platform.getInfo<CL_PLATFORM_NAME>() << endl;

	cout << "Select a " << default_platform.getInfo<CL_PLATFORM_NAME>()<< " device: " << endl;

	for (int i = 0; i < num_devices; i++)
	{
		cout << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << endl;
	}

	int selected_device_number;

	cin >> selected_device_number;

	cl::Device default_device = devices[selected_device_number];
	cout << "Using: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "on a " << default_device.getInfo<CL_DEVICE_NAME>() << endl;


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

			//Create the date_time structure and place it into the date_time vector
			struct tm date_time_struct;
			date_time_struct.tm_year = stoi(line_data[1]); // - 1900; //Year is the year from 1900 so 1900 has to be taken from the year given in the file
			date_time_struct.tm_mon = stoi(line_data[2]) - 1; //Month must be >= 0 and <= 11 so 1 must be taken from the given month value
			date_time_struct.tm_mday = stoi(line_data[3]);
			date_time_struct.tm_wday = 0;
			date_time_struct.tm_yday = 0;
			date_time_struct.tm_hour = stoi(line_data[4].substr(0, 2)); //The hour is the first two numbers in the time section of the file
			date_time_struct.tm_min = stoi(line_data[4].substr(2, 2)); //The minute is the two numbers following the hour in the time section
			date_time_struct.tm_sec = 0; //No second value is given so assumed to be 0

			date_time.push_back(date_time_struct);

			//Get the temperature and put it into the temperature structure
			temp.push_back(stof(line_data[5]));
		}
		data_file.close();
	}

	cout << date_time.size() << " Total Temperature Records \n";

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

	cl::Kernel addKernel(program, "addAll");

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
	vector<float> output(temp.size());
	size_t sizeOfOutput = output.size() * sizeof(float);

	size_t number_of_groups = temp.size() / local_size;

	//Create buffers
	cl::Buffer temperatures_buffer(context, CL_MEM_READ_WRITE, sizeOfInput);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, sizeOfOutput);

	//Create queue and copy vectors to device memory
	cl::CommandQueue queue(context);

	queue.enqueueWriteBuffer(temperatures_buffer, CL_TRUE, 0, sizeOfInput, &temp[0]);
	queue.enqueueFillBuffer(output_buffer, 0, 0, sizeOfOutput);

	//Execute kernel
	addKernel.setArg(0, temperatures_buffer);
	addKernel.setArg(1, output_buffer);
	addKernel.setArg(2, cl::Local(local_size * sizeof(float)));

	queue.enqueueNDRangeKernel(addKernel, cl::NullRange, cl::NDRange(temp.size()), cl::NDRange(local_size));

	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeOfOutput, &output[0]);

	float sum_p = output[0];

	float mean = sum_p / temp.size();

	//cout << sum_p << " Total temperature records." << endl;

	cout << "Average recorded temperature: " << mean << endl;
}

static void atomicAverage() {
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

	int num_platforms = platforms.size();

	cout << num_platforms << " platforms available on this machine:" << endl;

	cout << "Select Platform:" << endl;

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

			//Create the date_time structure and place it into the date_time vector
			struct tm date_time_struct;
			date_time_struct.tm_year = stoi(line_data[1]); // - 1900; //Year is the year from 1900 so 1900 has to be taken from the year given in the file
			date_time_struct.tm_mon = stoi(line_data[2]) - 1; //Month must be >= 0 and <= 11 so 1 must be taken from the given month value
			date_time_struct.tm_mday = stoi(line_data[3]);
			date_time_struct.tm_wday = 0;
			date_time_struct.tm_yday = 0;
			date_time_struct.tm_hour = stoi(line_data[4].substr(0, 2)); //The hour is the first two numbers in the time section of the file
			date_time_struct.tm_min = stoi(line_data[4].substr(2, 2)); //The minute is the two numbers following the hour in the time section
			date_time_struct.tm_sec = 0; //No second value is given so assumed to be 0

			date_time.push_back(date_time_struct);
			float __temp = stof(line_data[5]);
			//Get the temperature and put it into the temperature structure
			temp.push_back(int(__temp*100));
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

	cl::Kernel reduce_add(program, "reduce_add_4");

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

	printf("to div %i \n", temp.size());

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

	int num_platforms = platforms.size();

	cout << num_platforms << " platforms available on this machine:" << endl;

	cout << "Select Platform:" << endl;

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

			//Create the date_time structure and place it into the date_time vector
			struct tm date_time_struct;
			date_time_struct.tm_year = stoi(line_data[1]); // - 1900; //Year is the year from 1900 so 1900 has to be taken from the year given in the file
			date_time_struct.tm_mon = stoi(line_data[2]) - 1; //Month must be >= 0 and <= 11 so 1 must be taken from the given month value
			date_time_struct.tm_mday = stoi(line_data[3]);
			date_time_struct.tm_wday = 0;
			date_time_struct.tm_yday = 0;
			date_time_struct.tm_hour = stoi(line_data[4].substr(0, 2)); //The hour is the first two numbers in the time section of the file
			date_time_struct.tm_min = stoi(line_data[4].substr(2, 2)); //The minute is the two numbers following the hour in the time section
			date_time_struct.tm_sec = 0; //No second value is given so assumed to be 0

			date_time.push_back(date_time_struct);
			float __temp = stof(line_data[5]);
			//Get the temperature and put it into the temperature structure
			temp.push_back(int(__temp*100));
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

	size_t number_of_groups = temp.size() / local_size;

	//Create buffers
	cl::Buffer temperatures_buffer(context, CL_MEM_READ_WRITE, sizeOfInput);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, sizeOfOutput);

	//Create queue and copy vectors to device memory
	cl::CommandQueue queue(context);

	queue.enqueueWriteBuffer(temperatures_buffer, CL_TRUE, 0, sizeOfInput, &temp[0]);
	queue.enqueueFillBuffer(output_buffer, 0, 0, sizeOfOutput);

	//Execute kernel
	reduce_max.setArg(0, temperatures_buffer);
	reduce_max.setArg(1, output_buffer);
	reduce_max.setArg(2, cl::Local(local_size * sizeof(float)));

	queue.enqueueNDRangeKernel(reduce_max, cl::NullRange, cl::NDRange(temp.size()), cl::NDRange(local_size));

	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeOfOutput, &output[0]);

	float minimum = output[0];

	//printf("to div %i \n", temp.size());

	//float mean = sum_p / temp.size();

	cout << "Sum: " << minimum << endl;

	//cout << "Mean: " << mean << endl;

}

int main(int argc, char **argv) {

	while (true) {
		cout << "\n1. Average of temperatures. \n";
		cout << "2. Atomic Average. \n";
		cout << "3. Minimum and Maximum Temperatures. \n";
		cout << "q. Quit. \n";

		char choice;
		cin >> choice;

		if (choice == '1') {
			average();
		}
		if (choice == '2') {
			atomicAverage();
		}
		else if (choice == '3') {
			minMax();
		}
		else if (choice == 'q') {
			exit(0);
		}
	}
}