# PSNR calculating
from typing import Union
def calc_PSNR(inputs:torch.Tensor, targets:torch.Tensor,
              data_range:Union[int, float]=1.0) -> float:
    '''
    Args:
        inputs - the input tensor, should be of shape (N, C, H, W).
        targets - the target tensor, should be of shape (N, C, H, W).
        data_range - the data range of the given tensors, should be in ['255', '1.0'].
        reduction - the method to handle batched results. should be in ['none', 'sum', 'mean'].
    Returns:
        PSNR - the calculated results (mean value of the input batched sample).
    Reference:
        https://github.com/photosynthesis-team/piq/blob/master/piq/psnr.py
    '''
    # Constant for numerical stability, could guarantee accuracy in .5f
    assert inputs.shape == targets.shape, 'The shape of inputs is misaligned with that of outputs,' + \
        'please check their shape.'
    eps = 1e-10
    inputs = torch.clamp(inputs, 0, float(data_range))
    inputs, targets = inputs/float(data_range), targets/float(data_range)
    MSE = torch.mean((inputs - targets) ** 2, dim=[1, 2, 3])
    PSNR: torch.Tensor = - 10 * torch.log10(MSE + eps)
    return PSNR.mean(dim=0).item()


