from ratioestimator import *
from utils.simulator import GaussFunc
from eryn.prior import UniformTorch

import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from tqdm import tqdm

def main():

    # Number of simulations   
    N = 100000
    # Test performance of the method on the Gaussian shapes
    a_min, a_max = 0.5, 1.5 # amplitude
    b_min, b_max = -1.0, 1.0 # mean
    c_min, c_max = 0.1, 0.5 # standard deviation

    prior_a = UniformTorch(a_min, a_max)
    prior_b = UniformTorch(b_min, b_max)
    prior_c = UniformTorch(c_min, c_max)

    inputs_a = prior_a.sample((N,))
    inputs_a = inputs_a.view(-1, 1)
    
    inputs_b = prior_b.sample((N,))
    inputs_b = inputs_b.view(-1, 1)

    inputs_c = prior_c.sample((N,))
    inputs_c = inputs_c.view(-1, 1)  

    inputs = torch.cat((inputs_a, inputs_b, inputs_c),1)
   
    num = 100
    tvec = np.linspace(-1, 1, num)
    
    simulator = GaussFunc(tvec)
    outputs = simulator(inputs)
    
    dataset = TensorDataset(inputs, outputs)

    batch_size = 1024

    ones = torch.ones(batch_size, 1)
    zeros = torch.zeros(batch_size, 1) 

    hidden_sizes = [1024, 512, 256] 

    ratio_estimator = RatioEstimator(inputs.shape[1], outputs.shape[1], hidden_sizes) 
   
    optimizer = torch.optim.Adam(ratio_estimator.parameters(), lr=0.0001)
    criterion = torch.nn.BCELoss()

    num_epochs = 20

    losses = []

    for t in range(num_epochs):
      
        print('{}/'.format(str(t))+ '{}'.format(str(num_epochs))) 
        # Initialise dataloader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for batch in tqdm(iter(loader)):
            inputs, outputs = batch

            # Draw a sample from the joint p(inputs, outputs).               
            _, out = ratio_estimator(inputs, outputs)
            loss_joint = criterion(out, ones)
        
            # A sample from the product of marginals is drawn by simply shuffling along the batch axis.
            inputs = inputs[torch.randperm(inputs.size()[0])]
            _, out = ratio_estimator(inputs, outputs)
            loss_marginals = criterion(out, zeros)
        
            # Combine the losses and apply a backpropagation step
            optimizer.zero_grad()
            loss = loss_joint + loss_marginals
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

    plt.plot(losses)
    #plt.axvline(x_o, color="C0", label="True MAP")
    #plt.axvline(truth, color="red", label="Truth")
    plt.savefig('losses.png')

 
    
    # Save checkpoint at the end
    checkpoint_path = './checkpoint_gauss.pt'
    torch.save({'epoch': t,
                'model_state_dict': ratio_estimator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,}, checkpoint_path)


if __name__=="__main__":
    main()





