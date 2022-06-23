#ifndef NNET_CONV2D_RESOURCE_H_
#define NNET_CONV2D_RESOURCE_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_helpers.h"
#include <iostream>

namespace nnet {


// *************************************************
//       Helpers
// *************************************************

// A function calculating the elementwise product between two vectors/matrices (in this case, equivalent to a dot product)
template<typename data_T, typename res_T, int N>
inline void hadamard_product(const data_T a[], const data_T b[], res_T res[]) {
  #pragma unroll N
  for (int i = 0 ; i < N ; i++) {
    res[i] = (res_T) a[i] * b[i];
  }
}


// *************************************************
//       3x3 Kernels
// *************************************************

// Explicity transofrmed input (B'dB) needed for Winograd calculation, as explained by Lavin & Gray, 2015
template<typename data_T>
inline void winograd_transform_input_tile_3x3_kernel(const data_T I[16], data_T D[16]) {
    D[0] = I[0]-I[2]-I[8]+I[10];
    D[1] = I[1]+I[2]-I[9]-I[10];
    D[2] = -I[1]+I[2]+I[9]-I[10];
    D[3] = I[1]-I[3]-I[9]+I[11];

    D[4] = I[4]-I[6]+I[8]-I[10];
    D[5] = I[5]+I[6]+I[9]+I[10];
    D[6] = -I[5]+I[6]-I[9]+I[10];
    D[7] = I[5]-I[7]+I[9]-I[11];

    D[8] = -I[4]+I[6]+I[8]-I[10];
    D[9] = -I[5]-I[6]+I[9]+I[10];
    D[10] = I[5]-I[6]-I[9]+I[10];
    D[11] = -I[5]+I[7]+I[9]-I[11];

    D[12] = I[4]-I[6]-I[12]+I[14];
    D[13] = I[5]+I[6]-I[13]-I[14];
    D[14] = I[6]-I[5]+I[13]-I[14];
    D[15] = I[5]-I[7]-I[13]+I[15];
}

// Explicity transformed intermediate results, needed for obtaining convolution results, as explained by Lavin & Gray, 2015
template<typename data_T>
inline void winograd_transform_output_3x3_kernel(data_T Y[16], data_T Z[4]) {  
  Z[0] = Y[0]+Y[1]+Y[2]+Y[4]+Y[5]+Y[6]+Y[8]+Y[9]+Y[10];
  Z[1] = Y[1]-Y[2]-Y[3]+Y[5]-Y[6]-Y[7]+Y[9]-Y[10]-Y[11];
  Z[2] = Y[4]+Y[5]+Y[6]-Y[8]-Y[9]-Y[10]-Y[12]-Y[13]-Y[14];
  Z[3] = Y[5]-Y[6]-Y[7]-Y[9]+Y[10]+Y[11]+Y[15]-Y[13]+Y[14];
}

template<class data_T, class res_T, typename CONFIG_T>
void winograd_conv2d_3x3_kernel(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    const typename CONFIG_T::weight_t weights[CONFIG_T::n_filt * CONFIG_T::n_chan * CONFIG_T::filt_height * CONFIG_T::filt_width],
    const typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
) {
    assert(CONFIG_T::filt_height == 3 && CONFIG_T::filt_width == 3);
    assert(CONFIG_T::pad_left == CONFIG_T::pad_right);
    assert(CONFIG_T::pad_top == CONFIG_T::pad_bottom);

    // Reuse/untroll factors for channel and filter loops
    static constexpr int channel_loop_reuse = CONFIG_T::n_chan / CONFIG_T::reuse_factor;
    static constexpr int filter_loop_reuse = CONFIG_T::n_filt / CONFIG_T::reuse_factor;

    // Number of iterations, based on number of rows & columns, as well as padding
    // Explicitly calculated before the 'for' loop, as having it in the for loop fails to enter the loop
    // Likely cause is to do with compile-time casting - CONFG_T variables are unsigned
    static constexpr int height_itr = CONFIG_T::in_height + CONFIG_T::pad_bottom - 3;
    static constexpr int width_itr = CONFIG_T::in_width + CONFIG_T::pad_right - 3;

    #pragma loop_coalesce 2
    for (int row = -CONFIG_T::pad_top; row <= height_itr; row+=2) {
        std::cout << "Row loop?" << std::endl;
        for (int col = -CONFIG_T::pad_left; col <= width_itr ; col+=2) {            
            std::cout << "Col loop?" << std::endl;
            #pragma loop_coalesce 2
            // #pragma unroll channel_loop_reuse
            for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {   
                hls_register data_T D[16];
                hls_register data_T T[16];
                hls_register uint8_t p = 0;
                std::cout << "Iterating" << std::endl;
                #pragma unroll
                for (int r = row ; r < row+4 ; r++) {
                    #pragma unroll
                    for (int c = col ; c < col+4 ; c++) {
                        if (r < CONFIG_T::in_height && r >= 0 && c < CONFIG_T::in_width && c >= 0) {
                            T[p++] = data[r * CONFIG_T::in_width * CONFIG_T::n_chan + c * CONFIG_T::n_chan + channel];
                        } else {
                            T[p++] = 0;
                        }
                        std::cout << T[p-1] << std::endl;    
                    }
                }
                winograd_transform_input_tile_3x3_kernel<data_T>(T, D);

                // #pragma unroll filter_loop_reuse
                for (int filter = 0 ; filter < CONFIG_T::n_filt; filter++) {    
                    hls_register typename CONFIG_T::weight_t G[16];  
                    hls_register int filter_offset = 16*(CONFIG_T::n_chan*filter + channel); 
                    G[0] = weights[filter_offset];
                    G[1] = weights[filter_offset + 1];
                    G[2] = weights[filter_offset + 2];
                    G[3] = weights[filter_offset + 3];
                    G[4] = weights[filter_offset + 4];
                    G[5] = weights[filter_offset + 5];
                    G[6] = weights[filter_offset + 6];
                    G[7] = weights[filter_offset + 7];
                    G[8] = weights[filter_offset + 8];
                    G[9] = weights[filter_offset + 9];
                    G[10] = weights[filter_offset + 10];
                    G[11] = weights[filter_offset + 11];
                    G[12] = weights[filter_offset + 12];
                    G[13] = weights[filter_offset + 13];
                    G[14] = weights[filter_offset + 14];
                    G[15] = weights[filter_offset + 15];

                    hls_register data_T Y[16];
                    hls_register data_T Z[4];
                    hadamard_product<data_T, data_T, 16>(D, G, Y);
                    winograd_transform_output_3x3_kernel<data_T>(Y, Z);

                    if (channel==0) {
                        res[CONFIG_T::n_filt * ( (row+CONFIG_T::pad_top) * CONFIG_T::out_width + (col+CONFIG_T::pad_left) ) + filter] = (res_T) biases[filter];
                        res[CONFIG_T::n_filt * ( (row+CONFIG_T::pad_top) * CONFIG_T::out_width + (col+CONFIG_T::pad_left+1) ) + filter] = (res_T) biases[filter];
                        res[CONFIG_T::n_filt * ( (row+CONFIG_T::pad_top+1) * CONFIG_T::out_width + (col+CONFIG_T::pad_left) ) + filter] = (res_T) biases[filter];
                        res[CONFIG_T::n_filt * ( (row+CONFIG_T::pad_top+1) * CONFIG_T::out_width + (col+CONFIG_T::pad_left+1) ) + filter] = (res_T) biases[filter];
                    }

                    res[CONFIG_T::n_filt * ( (row+CONFIG_T::pad_top) * CONFIG_T::out_width + (col+CONFIG_T::pad_left) ) + filter] += (res_T) Z[0];
                    res[CONFIG_T::n_filt * ( (row+CONFIG_T::pad_top) * CONFIG_T::out_width + (col+CONFIG_T::pad_left+1) ) + filter] += (res_T) Z[1];
                    res[CONFIG_T::n_filt * ( (row+CONFIG_T::pad_top+1) * CONFIG_T::out_width + (col+CONFIG_T::pad_left) ) + filter] += (res_T) Z[2];
                    res[CONFIG_T::n_filt * ( (row+CONFIG_T::pad_top+1) * CONFIG_T::out_width + (col+CONFIG_T::pad_left+1) ) + filter] += (res_T) Z[3];
                }
            }
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_resource_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    const typename CONFIG_T::weight_t weights[CONFIG_T::n_filt * CONFIG_T::n_chan * CONFIG_T::filt_height * CONFIG_T::filt_width],
    const typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
) {
    winograd_conv2d_3x3_kernel<data_T, res_T, CONFIG_T>(data, res, weights, biases);

}

template<class data_T, typename CONFIG_T>
void im2col_2d_pointwise_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    data_T data_col[CONFIG_T::n_chan],
    const int row,
    const int col
) {
    hls_register uint index = 0;
    int input_row = -CONFIG_T::pad_top + row * CONFIG_T::stride_height;

    ChannelLoop:
    #pragma unroll
    for (uint channel = 0; channel < CONFIG_T::n_chan; channel++) {
        if (input_row < 0 || input_row >= CONFIG_T::in_height) {
            data_col[index++] = 0;
        } else {
            hls_register int input_col = -CONFIG_T::pad_left + col * CONFIG_T::stride_width;
            if (input_col >= 0 && input_col < CONFIG_T::in_width) {
                data_col[index++] = data[input_row * CONFIG_T::in_width * CONFIG_T::n_chan + input_col * CONFIG_T::n_chan + channel];
            } else {
                data_col[index++] = 0;
            }
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_2d_resource_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    const typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    const typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
) {
    assert(CONFIG_T::filt_height == 1 && CONFIG_T::filt_width == 1);

    hls_register data_T data_col[CONFIG_T::n_chan];
    hls_register res_T res_col[CONFIG_T::n_filt];

    HeightLoop:
    #pragma unroll
    for (uint i = 0; i < CONFIG_T::out_height; i++) {
        WidthLoop:
        #pragma unroll
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            im2col_2d_pointwise_cl<data_T, CONFIG_T>(data, data_col, i, j);
            dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
            FiltLoop:
            #pragma unroll
            for (int k = 0; k < CONFIG_T::n_filt; k++) {
                res[i * CONFIG_T::out_width * CONFIG_T::n_filt + j * CONFIG_T::n_filt + k] = res_col[k];
            }
        }
    }
}

}

#endif