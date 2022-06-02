#ifndef NNET_CONV1D_RESOURCE_H_
#define NNET_CONV1D_RESOURCE_H_

#include "nnet_common.h"
#include "nnet_dense.h"

namespace nnet {

template<class data_T, typename CONFIG_T>
void im2col_1d_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan], data_T data_col[CONFIG_T::filt_width * CONFIG_T::n_chan], const uint col) {
    hls_register uint index = 0;
    KernelLoop:
    #pragma unroll
    for (uint kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
        ChannelLoop:
        #pragma unroll
        for (uint channel = 0; channel < CONFIG_T::n_chan; channel++) {
            hls_register int index_data = (col*CONFIG_T::stride_width+kernel_col-CONFIG_T::pad_left) * CONFIG_T::n_chan + channel;
            if (index_data >= 0 && index_data < CONFIG_T::in_width*CONFIG_T::n_chan) {
                data_col[index++] = data[index_data];
            } else {
                data_col[index++] = 0;
            }
        }
    }
}

template<class data_T, typename CONFIG_T>
void im2col_1d_pointwise_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan], data_T data_col[CONFIG_T::n_chan], const uint col) {
    hls_register uint index = 0;
    ChannelLoop:
    #pragma unroll
    for (uint channel = 0; channel < CONFIG_T::n_chan; channel++) {
        hls_register int index_data = (col*CONFIG_T::stride_width-CONFIG_T::pad_left) * CONFIG_T::n_chan + channel;
        if (index_data >= 0 && index_data < CONFIG_T::in_width*CONFIG_T::n_chan) {
            data_col[index++] = data[index_data];
        } else {
            data_col[index++] = 0;
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_1d_resource_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_width * CONFIG_T::n_filt],
    const typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    const typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
) {
    
    hls_register data_T data_col[CONFIG_T::filt_width * CONFIG_T::n_chan];
    hls_register res_T res_col[CONFIG_T::n_filt];

    ColLoop:
    #pragma unroll
    for (uint i = 0; i < CONFIG_T::out_width; i++) {
        im2col_1d_cl<data_T, CONFIG_T>(data, data_col, i);
        dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
        FiltLoop:
        #pragma unroll
        for (uint j = 0; j < CONFIG_T::n_filt; j++) {
            res[i * CONFIG_T::n_filt + j] = res_col[j];
        }
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_resource_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_width * CONFIG_T::n_filt],
    const typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    const typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
) {
    assert(CONFIG_T::filt_width == 1);

    hls_register data_T data_col[CONFIG_T::n_chan];
    hls_register res_T res_col[CONFIG_T::n_filt];

    ColLoop:
    #pragma unroll
    for (uint i = 0; i < CONFIG_T::out_width; i++) {
        im2col_1d_pointwise_cl<data_T, CONFIG_T>(data, data_col, i);
        dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
        FiltLoop:
        #pragma unroll
        for (uint j = 0; j < CONFIG_T::n_filt; j++) {
            res[i * CONFIG_T::n_filt + j] = res_col[j];
        }
    }
}

}
#endif
