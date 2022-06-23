import numpy as np

from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.layers import Conv2D

class ApplyWinogradKernelTransformation(OptimizerPass):
    ''' 
    Transforms the weights of a 3x3 Conv2D kernel to a format suitable for Wingorad convolution
    For further information, refer to Lavin & Gray, 2015 - Fast Algorithms for Convolutional Neural Networks
    '''
    def match(self, node):
        node_matches = isinstance(node, (Conv2D))
        is_resource_strategy = node.get_attr('strategy', '').lower() == 'resource'
        weights_transformed = node.get_attr('_weights_transposed', False) == True
        already_transformed = node.get_attr('_winograd_transformation_applied', False) == True
        return node_matches and is_resource_strategy and weights_transformed and not already_transformed

    def transform(self, model, node):
        if isinstance(node, Conv2D):            
            weights = node.weights['weight'].data
            node.weights['weight'].data = np.zeros((weights.shape[0], weights.shape[1], 4, 4))
            
            G = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]])
            GT = np.array([[1, 0.5, 0.5, 0], [0, 0.5, -0.5, 0], [0, 0.5, 0.5, 1]])
            
            for filter in range(0, weights.data.shape[0]):
                for channel in range(0, weights.data.shape[1]):
                    node.weights['weight'].data[filter][channel] = np.matmul(np.matmul(G, weights[filter][channel]), GT)
                    node.weights['weight'].data_length = node.weights['weight'].data.size
        else:
            raise Exception('Unexpected layer {} with Winograd kernel optimizer'.format(node.class_name))

        node.set_attr('_winograd_transformation_applied', True)
        return False