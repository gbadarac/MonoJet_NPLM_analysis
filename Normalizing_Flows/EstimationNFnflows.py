#!/usr/bin/env python
# coding: utf-8

# Normalizing flow using nflows package and toy data 


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import rel_breitwigner
import torch
import os
import argparse
#instead of manually defining bijectors and distributions, 
#import necessary components from nflows
from nflows import distributions, flows, transforms
import nflows.transforms as transforms
from nflows.flows import Flow


# Add argument parsing for command line arguments
parser = argparse.ArgumentParser(description='Normalizing Flow Training Script')
parser.add_argument('--n_epochs', type=int, required=True, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for training')
parser.add_argument('--outdir', type=str, required=True, help='Output directory for saving results')

args = parser.parse_args()


#Setup: 
# - bkg: exponential falling distribution
# - signal: Breit-Wigner at certain mass 


# Generate background and signal data
n_bkg = 400000
n_sig = 20
bkg = np.random.exponential(scale=100.0, size=n_bkg)
sig = rel_breitwigner.rvs(450, size=n_sig)

'''
# Plot histogram
plt.hist([bkg, sig], bins=50, stacked=True)
plt.yscale('log')
plt.axvline(400.0, color='black', zorder=100)  # mass threshold or cut
plt.show()
'''

# Adding b-tagging information (a form of event classification)
bkg_btag = np.random.uniform(low=0.0, high=1.0, size=n_bkg)
sig_btag = np.random.normal(0.85, 0.05, n_sig)


#Combining energy and b-tagging score for both bkg and signal 
#Convert background coordinates to tensor
bkg_coord = np.column_stack((bkg_btag, bkg))  # Combine btag and bkg for training
bkg_coord = bkg_coord.astype('float32') #bkg coordinates converted to float32 for compatibility with python 

#sig_coord = np.column_stack((sig, sig_btag))
#sig_coord = sig_coord.astype('float32')

#sample points from target distribution 
y = torch.from_numpy(bkg_coord[:100000])  # Take the first 100,000 samples


# Note: the bkg distribution is the posterior/target distribution which the Normalizing Flow should learn to approximate. 

# Normalizing flow model:
# Set up simple normalizing flow with arbitrary inputs and outputs just to test 


# Define base distribution
# base distribution = prior distribution that the Normalizing Flow will transform 
base_distribution = distributions.StandardNormal(shape=(2,))
features=2 #dimensionality of the data being transformed.
# In this case: b-tagging score and background energy

# Define transformations (bijectors)
# you don't need to define a customed bijector anymore 

hidden_features = 5 #number of neurons in the NN 
num_blocks = 1 #number of layers in the NN 
transformations = transforms.MaskedAffineAutoregressiveTransform(features, hidden_features, num_blocks)

# The higher the number of hidden_features/num_blocks, the more expressive the transformation will be, 
# allowing it to capture more complex relationships in the data.

# The neural network basically has the base distribution values as inputs and gets to the parameters of the target distribution (via the network). 
# Then those parameters are inserted in the target distribution to get the ouputs in correspondence to the inputs. In this case, the neural network has 16 layers. 

# Using a neural network inside the transformations in normalizing flows does make the training loop "deeper" 
# in the sense that you're not just applying a single transformation but a series of transformations that are learned through the neural network.

# Setting up the normalizing flow and the training loop

#Create the flow
flow = Flow(transformations, base_distribution) #encapsules the entire flow model in a more structured way

#Training loop
#The training loop remains similar,
#but with flow.log_prob(y) directly calculating the log probability using the nflows implementation.

opt = torch.optim.Adam(flow.parameters(), args.learning_rate)

last_loss = np.inf
patience = 0


for idx in range(args.n_epochs):
    opt.zero_grad()

    # Minimize KL(p || q)
    loss = -flow.log_prob(y).mean()

    if idx % 1 == 0:
        print('epoch', idx, 'loss', loss)

    loss.backward()

    # Early stopping based on patience
    #patience mechanism keeps track of how many epochs have passed without improvement in loss. 
    #If the loss does not improve for 5 consecutive epochs, the training stops early to prevent overfitting.

    if loss > last_loss:
        patience += 1
    if patience >= 5:
        break
    last_loss = loss

    opt.step()

#Sampling: done using the provided methods from nflows
# Sample points from the base distribution
prior = base_distribution.sample(1000).numpy()  # Sample 1000 points with 2 features each

# Sample points from the trained flow
trained = flow.sample(1000).detach().numpy()  # Sample 1000 points with 2 features each

# Create output directory if it doesn't exist
os.makedirs(args.outdir, exist_ok=True)

# After creating the scatter plot
plt.scatter(prior[:, 0], prior[:, 1], color='gray', label='Base/Prior distribution')
plt.scatter(bkg_coord[:10000, 0], bkg_coord[:10000, 1], color='blue', label='Background/Target distribution')
plt.scatter(trained[:, 0], trained[:, 1], color='green', label='Trained distribution')
plt.xlabel("Latent b-tagging score")
plt.ylabel("Energy [GeV]")
plt.legend()

# Save the plot to the output directory
plot_path = os.path.join(args.outdir, f'plot.png')
plt.savefig(plot_path)

# Optionally, save other results like the final model state or loss values if needed
# torch.save(flow.state_dict(), os.path.join(args.outdir, 'flow_model.pth'))
