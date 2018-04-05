//a simple OpenCL kernel which adds two vectors A and B together into a third vector C
__kernel void add(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);
	printf("work item id = %d\n", id);

	if (id == 0) { //perform this part only once i.e. for work item 0
		printf("\nwork group size %d\n\n", get_local_size(0));
	}

	C[id] = A[id] + B[id];
}