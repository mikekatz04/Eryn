# A collection of the networks for the purposes of:
# approximating the ratio estimator,
# embedding the context.

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_inputs, 
                       num_hidden, 
                       num_outputs, 
                       act_func = nn.ReLU, 
                       activate_output = False):
        super(MLP, self).__init__()
        
        self.act_func = act_func
        self.first_layer = nn.Linear(num_inputs, num_hidden[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip(num_hidden[:-1], num_hidden[1:])])
        self.last_layer = nn.Linear(num_hidden[-1], num_outputs)
        self.activate_output = activate_output

    def forward(self, x):

        x = self.first_layer(x)
        x = self.act_func(x)

        for h_layer in self.hidden_layers:
            x = h_layer(x)
            x = self.act_func(x)

        x = self.last_layer(x)
        if self.activate_output:
            x = self.act_func(x)

        return x

