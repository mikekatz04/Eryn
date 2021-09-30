import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from nets import MLP

class RatioEstimator(nn.Module):
    '''
       Performs estimation of the likelihood-to-evidence ratio.
    '''
    def __init__(self, param_size, context_size, hidden_sizes):
        super(RatioEstimator, self).__init__()

        self.estimate =  MLP(param_size + context_size, hidden_sizes, 1, act_func = F.elu, activate_output = False)

    def forward(self, inputs, outputs):

        log_ratios = self.log_ratio(inputs, outputs)
        
        return log_ratios, log_ratios.sigmoid()
    
    def log_ratio(self, inputs, outputs):
 
        z = torch.cat([inputs, outputs], dim=1)
        
        return self.estimate(z) 


