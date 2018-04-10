__kernel void add(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);

	C[id] = A[id] + B[id];
}

__kernel void addAll(__global const float* input, __global float* output, __local float* kernel_mem) {

	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);

	kernel_mem[local_id] = input[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < local_size; i *= 2) {
		if (local_id % (i * 2) == false && ((local_id + i) < local_size)) {
			kernel_mem[local_id] += kernel_mem[local_id + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (local_id == false) {
		float add_val = kernel_mem[local_id];
		while (add_val != 0.0) {
			float old_val = atomic_xchg(&output[0], 0.0);
			add_val = atomic_xchg(&output[0], old_val + add_val);
		}
	}
}

__kernel void reduce_add_4(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0], scratch[lid]);
	}
}

__kernel void reduce_min(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			if (scratch[lid] > scratch[lid + 1]){
				scratch[lid] = scratch[lid + 1];
			}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_min(&B[0], scratch[lid]);
	}
}

__kernel void reduce_max(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			if (scratch[lid] < scratch[lid + 1]) {
				scratch[lid] = scratch[lid + 1];
			}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_max(&B[0], scratch[lid]);
	}
}