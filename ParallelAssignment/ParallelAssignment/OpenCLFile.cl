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

__kernel void reduce_max(__global const int* A, __global int* B, __local int* scratch) {
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

//__kernel void merge(__global float data_t * in, __global data_t * out, __local data_t * aux)
//{
//	int i = get_local_id(0); // index in workgroup
//	int wg = get_local_size(0); // workgroup size = block size, power of 2
//
//								// Move IN, OUT to block start
//	int offset = get_group_id(0) * wg;
//	in += offset; out += offset;
//
//	// Load block in AUX[WG]
//	aux[i] = in[i];
//	barrier(CLK_LOCAL_MEM_FENCE); // make sure AUX is entirely up to date
//
//								  // Now we will merge sub-sequences of length 1,2,...,WG/2
//	for (int length = 1;length<wg;length <<= 1)
//	{
//		data_t iData = aux[i];
//		uint iKey = getKey(iData);
//		int ii = i & (length - 1);  // index in our sequence in 0..length-1
//		int sibling = (i - ii) ^ length; // beginning of the sibling sequence
//		int pos = 0;
//		for (int inc = length;inc>0;inc >>= 1) // increment for dichotomic search
//		{
//			int j = sibling + pos + inc - 1;
//			uint jKey = getKey(aux[j]);
//			bool smaller = (jKey < iKey) || (jKey == iKey && j < i);
//			pos += (smaller) ? inc : 0;
//			pos = min(pos, length);
//		}
//		int bits = 2 * length - 1; // mask for destination
//		int dest = ((ii + pos) & bits) | (i & ~bits); // destination index in merged sequence
//		barrier(CLK_LOCAL_MEM_FENCE);
//		aux[dest] = iData;
//		barrier(CLK_LOCAL_MEM_FENCE);
//	}
//
//	// Write output
//	out[i] = aux[i];
//}