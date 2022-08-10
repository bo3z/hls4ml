#ifndef NNET_CONV2D_STREAM_H_
#define NNET_CONV2D_STREAM_H_

#include "nnet_conv_stream.h"
#include "nnet_types.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void conv_2d_cl(
    stream<data_T> &data,
    stream<res_T>  &res,
    const typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    const typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt] 
) {
    
    // Line buffer and kernel window
    hls_register static nnet::shift_reg<typename data_T::value_type, CONFIG_T::pad_left + CONFIG_T::in_width + CONFIG_T::pad_right> line_buffer[MAX(CONFIG_T::filt_height - 1, 1)][CONFIG_T::n_chan];
    hls_register static typename data_T::value_type kernel_window[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];

    // An array of length CONFIG_T::n_chan, with elements set to zero (padding for each channel)
    static const data_T padds(0);

    // Padding above input image
    PaddingTopHeight: 
    #pragma loop_coalesce 2
    for (int row = 0; row < CONFIG_T::pad_top; row++) {
        PaddingTopWidth: 
        for (int col = 0; col < CONFIG_T::pad_left + CONFIG_T::in_width + CONFIG_T::pad_right; col++) {
            compute_output_buffer<data_T, res_T, CONFIG_T>(padds, res, line_buffer, kernel_window, weights, biases);
        }
    }

    ReadInputHeight: 
    #pragma loop_coalesce 2
    for (int row = 0; row < CONFIG_T::in_height; row++) {
        // Input image left-side padding
        PaddingLeftWidth: 
        for (int col = 0; col < CONFIG_T::pad_left; col++) {
            compute_output_buffer<data_T, res_T, CONFIG_T>(padds, res, line_buffer, kernel_window, weights, biases);
        }
        
        // Read input image
        ReadInputWidth: 
        for (int col = 0; col < CONFIG_T::in_width; col++) {
            compute_output_buffer<data_T, res_T, CONFIG_T>(data.read(), res, line_buffer, kernel_window, weights, biases);
        }

        // Input image right-side padding
        PaddingRightWidth: 
        for (int col = 0; col < CONFIG_T::pad_right; col++) {
            compute_output_buffer<data_T, res_T, CONFIG_T>(padds, res, line_buffer, kernel_window, weights, biases);
        }
    }

    // Padding below input image
    PaddingBottomHeight: 
    #pragma loop_coalesce 2
    for (int row = 0; row < CONFIG_T::pad_bottom; row++) {
        PaddingBottomWidth: 
        for (int col = 0; col < CONFIG_T::pad_left + CONFIG_T::in_width + CONFIG_T::pad_right; col++) {
            compute_output_buffer<data_T, res_T, CONFIG_T>(padds, res, line_buffer, kernel_window, weights, biases);
        }
    }
}

#endif

}