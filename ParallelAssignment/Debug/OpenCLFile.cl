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
	if (!local_id) {
		float add_val = kernel_mem[local_id];
		while (add_val != 0.0) {
			float old_val = atomic_xchg(&output[0], 0.0);
			add_val = atomic_xchg(&output[0], old_val + add_val);
		}
	}
}