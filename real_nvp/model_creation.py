import torch

from real_nvp.coupling_layer import CouplingLayer
from real_nvp.gated_convnet import GatedConvNet
from real_nvp.image_flow import ImageFlow
from real_nvp.dequantization import Dequantization, VariationalDequantization
from real_nvp.multi_scale import SqueezeFlow, SplitFlow
from real_nvp.mask_utils import create_checkerboard_mask, create_channel_mask


def create_simple_flow(use_vardeq=True, device='cpu'):
    flow_layers = []
    if use_vardeq:
        vardeq_layers = [CouplingLayer(network=GatedConvNet(c_in=2, c_out=2, c_hidden=16),
                                       mask=create_checkerboard_mask(h=28, w=28, invert=(i%2==1)),
                                       c_in=1) for i in range(4)]
        flow_layers += [VariationalDequantization(var_flows=vardeq_layers)]
    else:
        flow_layers += [Dequantization()]
    
    for i in range(8):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=1, c_hidden=32),
                                      mask=create_checkerboard_mask(h=28, w=28, invert=(i%2==1)),
                                      c_in=1)]
        
    flow_model = ImageFlow(flow_layers).to(device)
    return flow_model


def create_multiscale_flow_from_config(
    config: dict
):
    flow_layers = []
    arch_config = config['architecture']
    hyperparams = config['hyperparameters']
    
    vardeq_layers = [
        CouplingLayer(
            network=GatedConvNet(
                c_in=arch_config['vardeq_layers']['CouplingLayer']['GatedConvNet']['c_in'], 
                c_out=arch_config['vardeq_layers']['CouplingLayer']['GatedConvNet']['c_out'], 
                c_hidden=arch_config['vardeq_layers']['CouplingLayer']['GatedConvNet']['c_hidden']
            ),
            mask=create_checkerboard_mask(
                h=arch_config['vardeq_layers']['CouplingLayer']['mask']['h'], 
                w=arch_config['vardeq_layers']['CouplingLayer']['mask']['w'], 
                invert=(i%2==1)
            ),
            c_in=arch_config['vardeq_layers']['CouplingLayer']['c_in']
        ) 
        for i in range(arch_config['vardeq_layers']['num_layers'])
    ]
    flow_layers += [VariationalDequantization(vardeq_layers)]
    
    flow_layers += [
        CouplingLayer(
            network=GatedConvNet(
                c_in=arch_config['low_layers']['CouplingLayer']['GatedConvNet']['c_in'],
                c_hidden=arch_config['low_layers']['CouplingLayer']['GatedConvNet']['c_hidden']
            ),
            mask=create_checkerboard_mask(
                h=arch_config['low_layers']['CouplingLayer']['mask']['h'], 
                w=arch_config['low_layers']['CouplingLayer']['mask']['w'], 
                invert=(i%2==1)),
            c_in=arch_config['low_layers']['CouplingLayer']['c_in']
        ) 
        for i in range(arch_config['low_layers']['num_layers'])
    ]
    flow_layers += [SqueezeFlow()]
    
    flow_layers += [
        CouplingLayer(
            network=GatedConvNet(
                c_in=arch_config['after_squeeze_layers']['CouplingLayer']['GatedConvNet']['c_in'], 
                c_hidden=arch_config['after_squeeze_layers']['CouplingLayer']['GatedConvNet']['c_hidden']
            ),
            mask=create_channel_mask(
                c_in=arch_config['after_squeeze_layers']['CouplingLayer']['mask']['c_in'], 
                invert=(i%2==1)
            ),
            c_in=arch_config['after_squeeze_layers']['CouplingLayer']['c_in']
        )
        for i in range(arch_config['after_squeeze_layers']['num_layers'])
    ]
    flow_layers += [SplitFlow(),
                    SqueezeFlow()]
    
    flow_layers += [
        CouplingLayer(
            network=GatedConvNet(
                c_in=arch_config['after_split_layers']['CouplingLayer']['GatedConvNet']['c_in'],
                c_hidden=arch_config['after_split_layers']['CouplingLayer']['GatedConvNet']['c_hidden']
            ),
            mask=create_channel_mask(
                c_in=arch_config['after_split_layers']['CouplingLayer']['mask']['c_in'],
                invert=(i%2==1)
            ),
            c_in=arch_config['after_split_layers']['CouplingLayer']['c_in']
        )
        for i in range(arch_config['after_split_layers']['num_layers'])
    ]

    flow_model = ImageFlow(
        flows=flow_layers,
        learning_rate=hyperparams['learning_rate'],
        gamma=hyperparams['gamma'],
        step_size=hyperparams['step_size'],
        example_input_array=hyperparams.get('example_input_array'),
        visualize_samples_shape=hyperparams.get('visualize_samples_shape'),
    )
    return flow_model


