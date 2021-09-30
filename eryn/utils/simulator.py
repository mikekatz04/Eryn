import torch
import numpy as np

class Simulator(torch.nn.Module):
    """ 
        This class implements the simulator of the forward model.
        In this implementation we assume that we have access to the 
        Simulator that can produce samples from the posterior distribution.
        TODO: We have to implement the case when we have access to the 'joint distribution',
        i.e. samples from the prior together with the correstonding realisation of the 
        data plus noise. This is the case that we will need to implement for the gravitational waves.

    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        """
            Defines the computation of the forward model at every call.
            Note: Should be overridden by all subclasses.
        """
        raise NotImplementedError

   
class GaussianSimulator(Simulator):

    def __init__(self):
        '''
           Implement Simulator for the Gaussian Distribution
           with unit variance and variable mean.
        '''
        super().__init__()

    def forward(self, inputs):
        inputs = inputs.view(-1, 1)
        return torch.randn(inputs.size(0), 1) + inputs


class GaussFunc(Simulator):

    def __init__(self, tvec):
        '''
          Implement Simulator that output the function of the Gaussian shape for a given set of parameetrs.
        '''
        super().__init__()

        self.tvec = torch.from_numpy(tvec).type(torch.FloatTensor)

    def forward(self, inputs):

        a = inputs[:,0].view(-1,1)
        b = inputs[:,1].view(-1,1)
        c = inputs[:,2].view(-1,1)
        
        f_x = a * torch.exp(-(torch.pow((self.tvec - b),2)) / (2.0 * torch.pow(c,2)))

        return f_x



