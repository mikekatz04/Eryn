import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.simulator import GaussFunc
from ratioestimator import *
from eryn.prior import UniformTorch

def main():

    # If GPU is available use GPU otherwise use CPU
    if torch.cuda.is_available():
        dev = "cuda:0"
        dtype = torch.cuda.FloatTensor
    else:
        dev = "cpu"
        dtype = torch.FloatTensor

    # Define the true values of the parameters

    # True parameters
    a_true = torch.from_numpy(np.array((1.0))).type(dtype) # amplitude
    b_true = torch.from_numpy(np.array((0.1))).type(dtype) # mean
    c_true = torch.from_numpy(np.array((0.3))).type(dtype) # standard deviation

    inputs_true = torch.stack((a_true, b_true, c_true)).view(1,-1)
   
    num = 100
    tvec = np.linspace(-1, 1, num)

    simulator = GaussFunc(tvec)
    outputs_true = simulator(inputs_true)

    # Scan the space of inputs and assign the same true output to them
    resolution = 100
    inputs_b_test = torch.linspace(-1.0, 1.0, resolution).view(-1,1)
    inputs_a_test = a_true.repeat(resolution, 1)
    inputs_c_test = c_true.repeat(resolution, 1)

    outputs_test = outputs_true.repeat(resolution, 1)
    inputs_test = torch.cat((inputs_a_test, inputs_b_test, inputs_c_test),1) 

    # Load the checkpoint
    hidden_sizes = [1024, 512, 256]
    ratio_estimator = RatioEstimator(inputs_test.shape[1], outputs_test.shape[1], hidden_sizes)
    optimizer = torch.optim.Adam(ratio_estimator.parameters(), lr=0.0001)

    checkpoint = torch.load('checkpoint_gauss.pt')
    ratio_estimator.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    t0 = checkpoint['epoch']
    loss = checkpoint['loss']

    # Calculate log posterior for all the points     
    log_ratios = ratio_estimator.log_ratio(inputs_test, outputs_test)
    print('log_ratios = ', log_ratios)   

    b_min, b_max = -1.0, 1.0 # mean
    prior_b = UniformTorch(b_min, b_max)
    log_prior = prior_b.log_prob(inputs_b_test).view(-1,1)
    print('log_prior = ', log_prior) 
    log_posterior =  log_ratios # + log_prior

    inputs_b_test_np = inputs_b_test.view(-1).detach().numpy()
    log_posterior_np = log_posterior.view(-1).detach().numpy()

    # Make plots

    print('inputs_b_test_np = ', inputs_b_test_np)
    print('log_posterior_np = ', log_posterior_np)

    plt.figure()
    plt.plot(inputs_b_test_np, log_posterior_np, lw=2, color="black")
    #plt.axvline(x_o, color="C0", label="True MAP")
    plt.axvline(0.1, color="red", label="Truth")
    #plt.legend()
    plt.savefig('log_posterior.png')

    plt.figure()
    plt.plot(inputs_b_test_np, np.exp(log_posterior_np), lw=2, color="black")
    #plt.axvline(x_o, color="C0", label="True MAP")
    #plt.axvline(truth, color="red", label="Truth")
    #plt.legend()
    plt.savefig('posterior.png')



if __name__=="__main__":
    main()


