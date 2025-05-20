#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

extern "C" {

__global__ void forward_hidden1(float* input, 
                              float* weights, 
                              float* output, 
                              int input_size, 
                              int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[i * hidden_size + idx];
        }
        // ReLU activation
        output[idx] = sum > 0.0f ? sum : 0.0f;
    }
}

__global__ void forward_hidden2(float* hidden1, 
                              float* weights, 
                              float* output, 
                              int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            sum += hidden1[i] * weights[i * hidden_size + idx];
        }
        // ReLU activation
        output[idx] = sum > 0.0f ? sum : 0.0f;
    }
}

__global__ void forward_output(float* hidden2, 
                             float* weights, 
                             float* output, 
                             int hidden_size, 
                             int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            sum += hidden2[i] * weights[i * output_size + idx];
        }
        
        if (idx < 2) {  // Movement actions (moveX, moveZ)
            // Tanh activation for bounded movement [-1, 1]
            output[idx] = tanhf(sum);
        } else {  // Shooting action
            // Sigmoid activation for shooting [0, 1]
            output[idx] = 1.0f / (1.0f + expf(-sum));
        }
    }
}

__global__ void forward_value(float* hidden2, 
                            float* weights, 
                            float* output, 
                            int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            sum += hidden2[i] * weights[i];
        }
        output[0] = sum;  // Linear activation for value
    }
}

__global__ void train(float* weights, 
                     float* gradients, 
                     float learning_rate, 
                     int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Gradient clipping
        float grad = gradients[idx];
        if (grad > 1.0f) grad = 1.0f;
        if (grad < -1.0f) grad = -1.0f;
        weights[idx] -= learning_rate * grad;
    }
}

} 