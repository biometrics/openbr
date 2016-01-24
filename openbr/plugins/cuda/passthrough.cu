__global__ void br_cuda_device_kernel() {

}

void br_cuda_device_wrapper() {
  br_cuda_device_kernel<<<1,1>>>();
}
