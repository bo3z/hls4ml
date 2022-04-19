import numpy as np
from copy import copy

from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.layers import register_layer
from hls4ml.backends.fpga.fpga_layers import PointwiseConv1D, PointwiseConv2D
from hls4ml.backends.quartus.passes.convolution_templates import Conv2DConfigTemplate, Conv2DFunctionTemplate, conv2d_config_template, conv_mult_config_template

pointwise_conv2d_function_template = 'nnet::pointwise_conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'

# TODO - Include streaming implementations once supported on Quartus
sepconv2d_include_list = ['nnet_utils/nnet_conv2d.h']

class PointwiseConv2DConfigTemplate(Conv2DConfigTemplate):
    def __init__(self):
        super(Conv2DConfigTemplate, self).__init__(PointwiseConv2D)
        self.template = conv2d_config_template
        self.mult_template = conv_mult_config_template

class PointwiseConv2DFunctionTemplate(Conv2DFunctionTemplate):
    def __init__(self):
        super(Conv2DFunctionTemplate, self).__init__(PointwiseConv2D, include_header=sepconv2d_include_list)
        self.template = pointwise_conv2d_function_template


def register_pointwise(backend):
    # Register the layer types to the layer map
    register_layer('PointwiseConv2D', PointwiseConv2D)

    # Register the optimization passes
    backend.register_pass('optimize_pointwise_conv', OptimizePointwiseConv)

    # Register template passes
    backend.register_template(PointwiseConv2DConfigTemplate)
    backend.register_template(PointwiseConv2DFunctionTemplate)

class OptimizePointwiseConv(OptimizerPass):
    def match(self, node):
        return node.class_name in ('Conv1D', 'Conv2D') and \
            node.get_attr('filt_height', 1) == 1 and \
            node.get_attr('filt_width') == 1

    def transform(self, model, node):
        dim = node.__class__.__name__[-2:] # '1D' or '2D'
        pw_node = model.make_node('PointwiseConv' + dim, node.name, copy(node.attributes), node.inputs.copy())
        if len(node.weights['weight'].data.shape) == 2: # This can happen if we assign weights of Dense layer to 1x1 Conv2D
            pw_node.weights['weight'].data = np.expand_dims(node.weights['weight'].data, axis=(0,1))
        pw_node.weights['bias'].data = node.weights['bias'].data
        model.replace_node(node, pw_node)
        
        return True
