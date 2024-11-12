#!/usr/bin/env python
# coding: utf-8

# Normalizing flow using nflows package and toy data 

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import rv_continuous
from scipy.stats import rel_breitwigner
import torch #Pytorch used to build and train ML models 
import flowtorch.bijectors as bij #contains different bijectors or transformation layers to build complex distributions
import flowtorch.distributions as dist #provides a way to define Normalizing Flows as a combination of a base distribtion and a set of transformations 
torch.cuda.is_available() #checking if a GPU is available for computations 

import sklearn
#instead of manually defining bijectors and distributions, 
#import necessary components from nflows
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import os
import argparse

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
bkg = np.random.exponential(scale=100.0,size=n_bkg)
sig = rel_breitwigner.rvs(450, size=n_sig)

# Adding b-tagging information (a form of event classification)
bkg_btag = np.random.uniform(low=0.0, high=1.0, size=n_bkg) #events in the bkg probably coming from a b-quark 
sig_btag = np.random.normal(0.85, 0.05, n_sig) #events in the signal probably coming from a b-quark

# Note: the bkg distribution is the posterior/target distribution which the Normalizing Flow should learn to approximate.

#Combining energy and b-tagging score for both bkg and signal 
bkg_coord = np.column_stack((bkg, bkg_btag))
#sig_coord = np.column_stack((sig, sig_btag))
#sig_coord = sig_coord.astype('float32')
#sig_coord_scaled = scaler.fit_transform(sig_coord)

# Define base distribution
base_distribution = torch.distributions.Independent(torch.distributions.Normal(torch.zeros(2), torch.ones(2)), 1) 

# Define target distribution
# Initialize the scaler 
scaler = StandardScaler()

#Scale the target distribution 
bkg_coord_scaled = scaler.fit_transform(bkg_coord)
bkg_coord_scaled = bkg_coord_scaled.astype('float32') #bkg coordinates converted to float32 for compatibility with python 

#Normalizing flow model:
# Set up simple normalizing flow with arbitrary inputs and outputs just to test 

# Define transformations (bijectors)
# you don't need to define a customed bijector anymore 
#transformations = transforms.MaskedAffineAutoregressiveTransform(features, args.hidden_features, args.num_blocks)
transformations = bij.Spline()

# The higher the number of hidden_features/num_blocks, the more expressive the transformation will be, 
# allowing it to capture more complex relationships in the data.
# The neural network basically has the base distribution values as inputs and gets to the parameters of the target distribution (via the network). 
# Then those parameters are inserted in the target distribution to get the ouputs in correspondence to the inputs. In this case, the neural network has 16 layers. 
# Using a neural network inside the transformations in normalizing flows does make the training loop "deeper" 
# in the sense that you're not just applying a single transformation but a series of transformations that are learned through the neural network.

# Setting up the normalizing flow and the training loop

#Create and initialize the flow
flow = dist.Flow(base_distribution, transformations) #encapsules the entire flow model in a more structured way

#Sample points from target distribution for training 
y = torch.from_numpy(bkg_coord_scaled[:100000])  # Take the first 100,000 samples
#y = torch.from_numpy(bkg_coord[:100000])  # Take the first 100,000 samples

'''
# Split the data into training and validation sets (e.g., 80% for training, 20% for validation)
train_size = int(0.8 * len(y))  # 80% for training
train_data = y[:train_size]
val_data = y[train_size:]
'''

#Training loop
opt = torch.optim.Adam(flow.parameters(), args.learning_rate) 


last_loss = np.inf
patience = 0
for idx in range(4001): #for each training step the following is computed
    opt.zero_grad()

    # Minimize KL(p || q)
    loss = -flow.log_prob(y).mean()

    if idx % 1 == 0:
        print('epoch', idx, 'loss', loss)

    loss.backward()

    #patience mechanism keeps track of how many epochs have passed without improvement in loss. 
    #If the loss does not improve for 5 consecutive epochs, the training stops early to prevent overfitting.
    
    if loss > last_loss:
        patience += 1
    if patience >=5:
        break
    last_loss = loss
    
    opt.step()


'''
train_losses=[]
val_losses=[]
min_loss=np.inf # Initialize minimum validation loss
min_loss_epoch=-1
patience_counter=0 # Initialize patience counter

for idx in range(args.n_epochs):
    flow.train() #set to taining mode 
    opt.zero_grad() #zero the gradients

    # Minimize KL(p || q)
    train_loss = -flow.log_prob(train_data).mean() #directly calculating the log probability using the nflows implementation
    # Append the loss for plotting later
    train_losses.append(train_loss.item())
    
    #Validation loss (analogously to training loss)
    flow.eval() #set to evaluation mode 
    val_loss = -flow.log_prob(val_data).mean()
    val_losses.append(val_loss.item())
    
    # Early stopping based on patience mechanism 
    # Patience mechanism keeps track of how many epochs have passed without improvement in loss. 
    # If the loss does not improve for 5 consecutive epochs, the training stops early to prevent overfitting.
    
    if val_loss < min_loss:
        min_loss = val_loss.item()
        min_loss_epoch = idx # Track the epoch where minimum loss occurs
        patience_counter = 0 # Reset patience counter if the loss improves
    else:
        patience_counter += 1
        if patience_counter >= 5: # If no improvement for 5 epochs, stop training
            print("Early stopping triggered")
            break
    
    # Use val_loss for early stopping because it provides a measure of how well the model generalizes to unseen data.
    # By monitoring val_loss instead of train_loss, we ensure that training does not continue if the model starts to overfit.
    # Early stopping based on val_loss helps prevent the model from improving solely on training data performance, which could lead to overfitting.   
    
    # Print progress every epoch
    if idx % 1 == 0:
        print(f"Epoch {idx}, Train Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}")

    train_loss.backward()
    opt.step()
'''
    
# Sample points from the trained flow
trained = flow.sample((1000,)).to('cpu').numpy()  # Sample 10000 points with 2 features each

# Sample points from the base distribution
# base distribution = prior distribution that the Normalizing Flow will transform 
prior = base_distribution.sample((1000,)).to('cpu').numpy()  # Sample 10000 points with 2 features each

'''
# Function to calculate KL divergence between target and trained distribution
def calculate_kl_divergence(target, trained, eps=1e-8):
    # Ensure target and trained are in probability space and avoid log(0) errors
    target = np.clip(target, eps, 1)  # Clip target values to avoid zero probabilities
    trained = np.clip(trained, eps, 1)  # Same for trained values
    # This prevents taking the logarithm of zero, which would lead to undefined (NaN) values and numerical instability in the KL divergence calculation. 
    # By clipping to this small positive value, it ensures no probability is exactly zero.

    # Convert numpy arrays to PyTorch tensors
    p_target = torch.from_numpy(target).float()
    q_trained = torch.from_numpy(trained).float().log()  # q_trained should be in log space

    # Calculate KL divergence
    kl_divergence = torch.nn.functional.kl_div(q_trained, p_target, reduction='batchmean')
    return kl_divergence.item()

# Calculate KL divergence
kl_div = calculate_kl_divergence(bkg_coord_scaled[:10000], trained)
#kl_div = calculate_kl_divergence(bkg_coord[:10000], trained)
'''

# Create output directory if it doesn't exist
os.makedirs(args.outdir, exist_ok=True)

# After creating the scatter plot
#plt.scatter(bkg_coord[:10000, 0], bkg_coord[:10000, 1], color='blue', label='Background/Target distribution')
plt.scatter(prior[:, 0], prior[:, 1], color='gray', label='Base/Prior distribution')
plt.scatter(bkg_coord_scaled[:10000, 0], bkg_coord_scaled[:10000, 1], color='blue', label='Background/Target distribution')
plt.scatter(trained[:, 0], trained[:, 1], color='green', label='Trained distribution')
plt.ylabel("Latent b-tagging score")
plt.xlabel("Energy [GeV]")
plt.title("Scatter Plot of Scaled Distributions: Target, Prior and Trained")
plt.legend(loc='upper left')

# Display hidden_features, num_blocks, and KL divergence in the plot
#text_str = f"n_epochs: {args.n_epochs}\nlearning_rate: {args.learning_rate}\nKL Divergence: {kl_div:.4f}"
text_str = f"n_epochs: {args.n_epochs}\nlearning_rate: {args.learning_rate}"
plt.text(0.6, 0.95, text_str, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.7))

# Save the plot to the output directory
scatter_name = f"scatter.png"
scatter_path = os.path.join(args.outdir, scatter_name)
plt.savefig(scatter_path)

'''
# Plot training and validation loss per epoch
# Training loss: measures how well your model is fitting the target distribution
# Validation loss: measures how well your model generalizes to unseen data fro the same target disribution

plt.figure()
plt.plot(train_losses, label="Training Loss", color='blue') #training loss 
plt.plot(val_losses, label="Validation Loss", color='red', linestyle='--') #validation loss 
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Epoch")
plt.legend()

# Display the minimum loss and the corresponding epoch in the plot
text_str = f"n_epochs: {args.n_epochs}\nlearning_rate: {args.learning_rate}\nKL Divergence: {kl_div:.4f}\nMin Loss: {min_loss:.4f} at Epoch {min_loss_epoch}"
plt.text(0.6, 0.95, text_str, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.7))

# Save the training loss plot
loss_name = f"loss.png"
loss_plot_path = os.path.join(args.outdir, loss_name)
plt.savefig(loss_plot_path)
'''