__kernel void addOneToInput(__global const float* A, __global float* C) {
	int id = get_global_id(0);
	C[id] = A[id] + 1.0;
}

__kernel void addAll(__global const float* input, __global float* output, __local float* local_cache) {
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);

	//Copy values into local memory
	local_cache[local_id] = input[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < local_size; i *= 2) {
		if (!(local_id % (i * 2)) && ((local_id + i) < local_size)) {
			local_cache[local_id] += local_cache[local_id + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (!local_id) {
		float add_val = local_cache[local_id];
		while (add_val != 0.0) {
			float old_val = atomic_xchg(&output[0], 0.0);
			add_val = atomic_xchg(&output[0], old_val + add_val);
		}
	}
}

__kernel void HS_Scan(__global const float* input, __global float* output, __local float* local_cache, __local float* local_out) {
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);

	__local float *temp;

	local_cache[local_id] = input[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < local_size; i *= 2) {
		if (local_id >= i)
			local_out[local_id] = local_cache[local_id] + local_cache[local_id - i];
		else
			local_out[local_id] = local_cache[local_id];

		barrier(CLK_LOCAL_MEM_FENCE);

		temp = local_out;
		local_out = local_cache;
		local_cache = temp;
	}

	output[id] = local_cache[local_id];
}

__kernel void B_Scan(__global const float* input, __global float* output, __local float* local_cache)
{
	// Implementation: nehalemlabs.net/prototype/blog/2014/06/23/parallel-programming-with-opencl-and-python-parallel-scan/
	// developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html

	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);
	int offset = 1;


	//Copy values into local memory
	local_cache[2 * local_id] = input[2 * id];
	local_cache[2 * local_id + 1] = input[2 * id + 1];
	barrier(CLK_LOCAL_MEM_FENCE);

	/* UPSWEEP */
	for (int d = local_size >> 1; d > 0; d >>= 1) {
		if (local_id < d) {
			int i = offset * (2 * local_id + 1) - 1;
			int j = offset * (2 * local_id + 2) - 1;

			local_cache[j] += local_cache[i];
		}

		offset <<= 1;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/* DOWNSWEEP */
	if (local_id == 0) {
		local_cache[local_size - 1] = 0; //Set rightmost value to identity operator (in this case 0)
	}
	for (int d = 1; d < local_size + (local_size / 2); d *= 2) { //local_size+(local_size/2) gets it to do the final step
		if (local_id <= d) {

			int i = offset * (2 * local_id + 1) - 1;
			int j = offset * (2 * local_id + 2) - 1;

			float temp = local_cache[j];

			local_cache[j] += local_cache[i];
			local_cache[i] = temp;
		}
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	output[2 * id] = local_cache[2 * local_id];
	output[2 * id + 1] = local_cache[2 * local_id + 1];
}

__kernel void blelloch_max(__global const float* input, __global float* output, __local float* local_cache)
{
	// Implementation: nehalemlabs.net/prototype/blog/2014/06/23/parallel-programming-with-opencl-and-python-parallel-scan/
	// developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html

	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);
	int offset = 1;


	//Copy values into local memory
	local_cache[2 * local_id] = input[2 * id];
	local_cache[2 * local_id + 1] = input[2 * id + 1];
	barrier(CLK_LOCAL_MEM_FENCE);

	/* UPSWEEP */
	for (int d = local_size >> 1; d > 0; d >>= 1) {
		if (local_id < d) {
			int i = offset * (2 * local_id + 1) - 1;
			int j = offset * (2 * local_id + 2) - 1;

			local_cache[j] = max(local_cache[i], local_cache[j]);
		}

		offset <<= 1;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/* DOWNSWEEP */
	if (local_id == 0) {
		local_cache[local_size - 1] = 0; //Set rightmost value to identity operator (in this case 0)
	}
	for (int d = 1; d < local_size + (local_size / 2); d *= 2) { //local_size+(local_size/2) gets it to do the final step
		if (local_id <= d) {

			int i = offset * (2 * local_id + 1) - 1;
			int j = offset * (2 * local_id + 2) - 1;

			float temp = local_cache[j];

			local_cache[j] = max(local_cache[i], local_cache[j]);
			local_cache[i] = temp;
		}
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	output[2 * id] = local_cache[2 * local_id];
	output[2 * id + 1] = local_cache[2 * local_id + 1];
}

__kernel void odd_even_sort_ascending(__global const float* input, __global int* output, __local float* local_cache) {
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);
	int global_size = get_global_size(0);

	local_cache[local_id] = input[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < local_size; i++) {
		if ((!(local_id % 2 == 0) && i % 2 == 0) && local_id - 1 >= 0) {

			float temp = local_cache[local_id - 1];
			float min_val = min(local_cache[local_id], local_cache[local_id - 1]);

			if (min_val != local_cache[local_id - 1]) {
				local_cache[local_id - 1] = min_val;
				local_cache[local_id] = temp;
			}
		}
		else if ((!(local_id % 2 == 0) && i % 2 != 0) && local_id + 1 < local_size) {
			float min_val = min(local_cache[local_id], local_cache[local_id + 1]);

			if (min_val == local_cache[local_id + 1]) {
				float temp = local_cache[local_id];
				local_cache[local_id] = min_val;
				local_cache[local_id + 1] = temp;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	output[id] = (int)local_cache[local_id] * 1000;
	barrier(CLK_GLOBAL_MEM_FENCE);

	/* ATOMIC FUNCTION TO SORT GLOBALLY */
}