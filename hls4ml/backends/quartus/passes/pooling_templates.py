
from hls4ml.model.layers import Pooling1D, Pooling2D, GlobalPooling1D, GlobalPooling2D
from hls4ml.backends.template import LayerConfigTemplate, FunctionCallTemplate

# Pooling templates
# TODO - Expand this, similar to Vivado, once Conv1D / streaming is supported
pooling2d_config_template = """struct config{index} : nnet::pooling2d_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_filt = {n_filt};
    static const unsigned stride_height = {stride_height};
    static const unsigned stride_width = {stride_width};
    static const unsigned pool_height = {pool_height};
    static const unsigned pool_width = {pool_width};

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const nnet::Pool_Op pool_op = nnet::{pool_op};
    static const unsigned reuse = {reuse};
    typedef {accum_t.name} accum_t;
}};\n"""

pooling2d_function_template = 'nnet::pooling2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'

pooling_include_list = ['nnet_utils/nnet_pooling.h']

class PoolingConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Pooling2D)
        self.templates = {
            'Pooling2D': pooling2d_config_template,
        }

    def format(self, node):
        params = self._default_config_params(node)
        return self.templates[node.class_name].format(**params)

class PoolingFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Pooling2D, include_header=pooling_include_list)
        self.templates = {
            'Pooling2D': pooling2d_function_template,
        }

    def format(self, node):
        params = self._default_function_params(node)
        params['data_format'] = 'cf' if node.get_attr('data_format') == 'channels_first' else 'cl'
        return self.templates[node.class_name].format(**params)
