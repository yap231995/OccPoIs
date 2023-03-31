import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import re

def batchNormToDense(bnlayer):
    gamma,beta,mean,std = bnlayer.get_weights()
    #from what i can tell in the code std is actually variance
    epsilon = bnlayer.epsilon
    # print("bnlayer.input.shape", bnlayer.input.shape)
    featureDimension = gamma.shape[0]
    W = np.zeros(  (featureDimension,featureDimension)   )
    B = np.zeros(  (featureDimension,)  )
    for i in range( featureDimension ):
        denom_i = np.sqrt(std[i]) + epsilon
        W[i,i] = float( gamma[i]/ denom_i )
        B[i]   = -1*(mean[i]*gamma[i]/float(denom_i)) + beta[i]
    return Dense(featureDimension, weights = [ W, B ] )


def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
    print("START network_dict:", network_dict)
    # Set the input layers of each layer
    for layer in model.layers:
        print("layer._outbound_nodes", layer._outbound_nodes)
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                if layer_name not in network_dict['input_layers_of'][layer_name]:
                    network_dict['input_layers_of'][layer_name].append(layer.name)

    print("2 network_dict:", network_dict)
    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    print("3 network_dict:", network_dict)
    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        # if re.match(layer_regex, layer.name):
        if layer_regex == layer.name:
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory
            if insert_layer_name:
                new_layer._name = insert_layer_name
            else:
                new_layer._name = '{}_{}'.format(layer._name,
                                                new_layer._name)
            # print("x.shape", x.shape)
            x = new_layer(x)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer._name,
                                                            layer._name, position))
            if position == 'before':
                x = layer(x)
        else:
            # print("layer:", layer)
            # print("layer_input:", layer_input)
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        print("4 network_dict:", network_dict)
        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)