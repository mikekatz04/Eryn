import torch

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

