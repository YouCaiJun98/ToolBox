import torch
import torch.nn as nn
def net_device(module: nn.Module) -> torch.device:
    r"""
    Get current device of the network, assuming all weights of the network are on the same device.

    Args:
        module (nn.Module): The network.

    Returns:
        torch.device: The device.

    Examples::
      >>> module = nn.Conv2d(3, 3, 3)
      >>> device = net_device(module) # "cpu"
    """
    if isinstance(module, nn.DataParallel):
        module = module.module
    parameters = module._parameters
    if 'weight' in parameters:
        return parameters['weight'].device
    for submodule in module.children():
        parameters = submodule._parameters
        if "weight" in parameters:
            return parameters["weight"].device
        device = net_device(submodule) 
        if device:
            return device
