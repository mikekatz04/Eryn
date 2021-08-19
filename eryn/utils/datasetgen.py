import torch
from torch.utils.data import Dataset

import Simulator


class DatasetSimulator(Dataset):

    def __init__(self, simulator, prior, size):
    '''
       Implements Dataset which is required for Dataloader.
       This Dataset at each run samples new batch of data using the Simulator. 
       Inputs:
           simulator -- forward simulator for data genertaion
           prior     -- prior parameter distribution
           size      -- max size of the dataset (i.e. batch_size*number_of_iterations)
    '''  
        super().__init__()

        self.prior = prior
        self.simulator = simulator
        self.size = size
      
    def __getitem__(self, index):
        """
           Generates one sample of data which has a size defined by batch_size.
        """
        passed = False
        while not passed:
            try:
                # Not sure if this is correctly implemented
                print('index = ', index)
                inputs = self.prior.sample(torch.Size(index.shape[0])).unsqueeze(0)
                outputs = self.simulator(inputs)
                passed = True
            except Exception as e:
                print(e)

        return inputs, outputs

    def __len__(self):
        """
           Denotes the total numer of samples.
        """
        return self.size
