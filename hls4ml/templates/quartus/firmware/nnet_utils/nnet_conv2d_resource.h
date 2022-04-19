#ifndef NNET_CONV2D_RESOURCE_H_
#define NNET_CONV2D_RESOURCE_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_helpers.h"

namespace nnet {

template<class data_T, typename CONFIG_T>
void im2col_2d_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    data_T data_col[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
    const int row,
    const int col

) {
    hls_register uint index = 0;
    #pragma unroll
    for (uint kernel_row = 0; kernel_row < CONFIG_T::filt_height; kernel_row++) {
        hls_register int input_row = -CONFIG_T::pad_top + kernel_row * CONFIG_T::dilation_height + row * CONFIG_T::stride_height;
        #pragma unroll
        for (uint kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
            hls_register int input_col = -CONFIG_T::pad_left + kernel_col * CONFIG_T::dilation_width + col * CONFIG_T::stride_width;
            #pragma ii 1
            for (uint channel = 0; channel < CONFIG_T::n_chan; channel++) {
                if (input_row >= 0 && input_row < CONFIG_T::in_height && input_col >= 0 && input_col < CONFIG_T::in_width) {
                    data_col[index++] = data[input_row * CONFIG_T::in_width * CONFIG_T::n_chan + input_col * CONFIG_T::n_chan + channel];
                } else {
                    data_col[index++] = 0;
                }
            }
        }
    }
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
void conv_2d_resource_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    const typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    const typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
) {
    
    hls_register data_T data_col[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    hls_register res_T res_col[CONFIG_T::n_filt];
    
    HeightLoop: 
        #pragma unroll
        for (uint i = 0; i < CONFIG_T::out_height; i++) {
        WidthLoop: 
            #pragma unroll
            for (uint j = 0; j < CONFIG_T::out_width; j++) {
            im2col_2d_cl<data_T, CONFIG_T>(data, data_col, i, j);
            dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
            FiltLoop: 
                #pragma unroll
                for (uint k = 0; k < CONFIG_T::n_filt; k++) {
                    res[i * CONFIG_T::out_width * CONFIG_T::n_filt + j * CONFIG_T::n_filt + k] = res_col[k];
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