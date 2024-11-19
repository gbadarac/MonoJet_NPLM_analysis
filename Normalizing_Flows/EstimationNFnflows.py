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
import sklearn
#instead of manually defining bijectors and distributions, 
#import necessary components from nflows
from sklearn.preprocessing import StandardScaler

from nflows.distributions.normal import StandardNormal
from nflows import distributions, flows, transforms
import nflows.transforms as transforms
from nflows.flows import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add argument parsing for command line arguments
parser = argparse.ArgumentParser(description='Normalizing Flow Training Script')
parser.add_argument('--n_epochs', type=int, required=True, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for training')
parser.add_argument('--batch_size', type=int, required=False, default=1024, help='Batch size for training')
parser.add_argument('--outdir', type=str, required=True, help='Output directory for saving results')
parser.add_argument('--num_layers', type=int, required=True, help='Number of flow layers (each flow layer contains several transformations/blocks)')
parser.add_argument('--num_blocks', type=int, required=True, help='Number of transformations/blocks in each flow layer')
parser.add_argument('--hidden_features', type=int, required=True, help='Number of neurons in the NN inside each transformation/block')
parser.add_argument('--num_bins', type=int, required=True, help='Number of network parameters for each layer of spline transformations')
args = parser.parse_args()

#Setup: 
# - bkg: exponential falling distribution
# - signal: Breit-Wigner at certain mass 

# Generate background and signal data
n_bkg = 400000
n_sig = 20
bkg = np.random.exponential(scale=100.0, size=n_bkg)
sig = rel_breitwigner.rvs(450, size=n_sig)

# Adding b-tagging information (a form of event classification)
bkg_btag = np.random.uniform(low=0.0, high=1.0, size=n_bkg)
sig_btag = np.random.normal(0.85, 0.05, n_sig)

num_features=2 #dimensionality of the data being transformed.
# In this case: b-tagging score and background energy

# Note: the bkg distribution is the posterior/target distribution which the Normalizing Flow should learn to approximate.

#Combining energy and b-tagging score for both bkg and signal 
bkg_coord = np.column_stack((bkg_btag, bkg))  # Combine btag and bkg for training

#Initialize the scaler 
scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(-1,1))

#Scale the target distribution to help the model to converge faster 
bkg_coord_scaled = scaler.fit_transform(bkg_coord)

bkg_coord_scaled = bkg_coord_scaled.astype('float32') #bkg coordinates converted to float32 for compatibility with python 

#sig_coord = np.column_stack((sig, sig_btag))
#sig_coord = sig_coord.astype('float32')
#sig_coord_scaled = scaler.fit_transform(sig_coord)

# Define base distribution
base_distribution = distributions.StandardNormal(shape=(num_features,))

#Normalizing flow model:
# Set up simple normalizing flow with arbitrary inputs and outputs just to test 

# Define transformations (bijectors)
#transformations = transforms.MaskedAffineAutoregressiveTransform(features, args.hidden_features, args.num_blocks)

num_context=0

def make_flow(num_features,num_context, perm=True):
    base_dist = distributions.StandardNormal(shape=(num_features,))
    #base_dist = MultivariateScaledNormal(num_features, scale=3.0)

    transforms = []
    if num_context == 0:
        num_context = None
    for i in range(args.num_layers):
        transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=num_features,
                                                                                context_features=num_context,
                                                                                hidden_features=args.hidden_features,
                                                                                num_bins=args.num_bins,
                                                                                num_blocks=args.num_blocks,
                                                                                tail_bound=10.0, #range over which the spline trasnformation is defined 
                                                                                tails='linear',
                                                                                dropout_probability=0.2,
                                                                                use_batch_norm=False))
        if i < args.num_layers - 1 and perm:
            transforms.append(ReversePermutation(features=num_features)) #Shuffles feature order to increase expressivity
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow

# The higher the number of hidden_features/num_blocks, the more expressive the transformation will be, 
# allowing it to capture more complex relationships in the data.
# The neural network basically has the base distribution values as inputs and gets to the parameters of the target distribution (via the network). 
# Then those parameters are inserted in the target distribution to get the ouputs in correspondence to the inputs. In this case, the neural network has 16 layers. 
# Using a neural network inside the transformations in normalizing flows does make the training loop "deeper" 
# in the sense that you're not just applying a single transformation but a series of transformations that are learned through the neural network.

# Setting up the normalizing flow and the training loop

#Sample points from target distribution for training 
y = torch.from_numpy(bkg_coord_scaled[:100000])  # Take the first 100,000 samples

# Split the data into training and validation sets (e.g., 80% for training, 20% for validation)
train_size = int(0.8 * len(y))  # 80% for training
train_data = y[:train_size]
val_data = y[train_size:]

#Create and initialize the flow
#flow = Flow(transformations, base_distribution) #encapsules the entire flow model in a more structured way
flow = make_flow(num_features, num_context, perm=True)

#Training loop
opt = torch.optim.Adam(flow.parameters(), args.learning_rate)

train_dataset = TensorDataset(train_data)
val_dataset = TensorDataset(val_data)

#create loaders 
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

# Define scheduler
scheduler = CosineAnnealingLR(optimizer=opt, T_max=args.n_epochs, eta_min=1e-3)  # eta_min: minimum LR, T_max: total number of epochs for one cosine cycle 
#scheduler = OneCycleLR(optimizer=opt, max_lr=5e-4, steps_per_epoch=len(train_loader), epochs=args.n_epochs)  

train_losses=[]
val_losses=[]

min_loss=np.inf # Initialize minimum validation loss
min_loss_epoch=-1
patience_counter=0 # Initialize patience counter

for idx in range(args.n_epochs):
    flow.train() #set to taining mode 
    total_train_loss=0.0
    
    #loop over training 
    for batch in train_loader:
        batch_data = batch[0]
        opt.zero_grad() #zero the gradients and make predictions for a set of inputs 
        
        # Minimize KL(p || q), i.e. calculate the loss
        train_loss = -flow.log_prob(batch_data).mean() #calculating the log probability for batch
        train_loss.backward() 
        opt.step() #model updates its parameters (weights) 
        total_train_loss += train_loss.item() * batch_data.size(0) #accumulate batch loss 

    #Calculate average training loss for the current epoch
    avg_train_loss = total_train_loss / len(train_data)
    # Append the loss for plotting later
    train_losses.append(avg_train_loss)
    
    #Validation loss (analogously to training loss)
    flow.eval() #set to evaluation mode 
    total_val_loss=0.0
    
    #loop over validation 
    with torch.no_grad():
        for val_batch in val_loader:
            val_batch_data = val_batch[0]
            
            # Minimize KL(p || q)
            val_loss = -flow.log_prob(val_batch_data).mean() #calculating the log probability for validation
            total_val_loss += val_loss.item() * val_batch_data.size(0) #accumulate batch loss 
    
    #Calculate average validation loss for the current epoch
    avg_val_loss = total_val_loss / len(val_data) 
    # Append the loss for plotting later
    val_losses.append(avg_val_loss)
    
    # Scheduler step for CosineAnnealingLR
    scheduler.step() #adjusts the learning rate at each epoch based on cosine formula 

    # Early stopping based on patience mechanism 
    # Patience mechanism keeps track of how many epochs have passed without improvement in loss. 
    # If the loss does not improve for 5 consecutive epochs, the training stops early to prevent overfitting.

    if avg_val_loss < min_loss: #compare the epoch-level validation loss to min_loss
        min_loss = avg_val_loss
        min_loss_epoch = idx # Track the epoch where minimum loss occurs
        patience_counter = 0 # Reset patience counter if the loss improves
    else:
        patience_counter += 1
        if patience_counter >= 10: # If no improvement for 5 epochs, stop training
            print("Early stopping triggered")
            break
    
    # Use val_loss for early stopping because it provides a measure of how well the model generalizes to unseen data.
    # By monitoring val_loss instead of train_loss, we ensure that training does not continue if the model starts to overfit.
    # Early stopping based on val_loss helps prevent the model from improving solely on training data performance, which could lead to overfitting.   

    # Print progress every epoch
    if idx % 1 == 0:
        print(f"Epoch {idx}, Avg Train Loss: {avg_train_loss:.4f}, Avg Validation Loss: {avg_val_loss:.4f}")
    
# Sample points from the trained flow
trained = flow.sample(10000).detach().numpy()  # Sample 10000 points with 2 features each

# Sample points from the base distribution
prior = base_distribution.sample(10000).numpy()  # Sample 10000 points with 2 features each

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

# Create output directory if it doesn't exist
os.makedirs(args.outdir, exist_ok=True)

# After creating the scatter plot
plt.scatter(prior[:, 0], prior[:, 1], color='gray', label='Base/Prior distribution')
plt.scatter(bkg_coord_scaled[:10000, 0], bkg_coord_scaled[:10000, 1], color='blue', label='Background/Target distribution')
plt.scatter(trained[:, 0], trained[:, 1], color='green', label='Trained distribution')
plt.xlabel("Latent b-tagging score")
plt.ylabel("Energy [GeV]")
plt.title("Scatter Plot of Scaled Distributions: Target, Prior and Trained")
plt.legend(loc='lower left')

# Display hidden_features, num_blocks, and KL divergence in the plot
text_str = f"learning_rate: {args.learning_rate}\nnum_layers: {args.num_layers}\nnum_blocks: {args.num_blocks}\nhidden_features: {args.hidden_features}\nnum_bins: {args.num_bins}\nn_epochs: {args.n_epochs}\nKL Divergence: {kl_div:.4f}"
plt.text(0.6, 0.95, text_str, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.7))

# Save the plot to the output directory
scatter_name = f"scatter.png"
scatter_path = os.path.join(args.outdir, scatter_name)
plt.savefig(scatter_path)

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
text_str = f"learning_rate: {args.learning_rate}\nnum_layers: {args.num_layers}\nnum_blocks: {args.num_blocks}\nhidden_features: {args.hidden_features}\nnum_bins: {args.num_bins}\nn_epochs: {args.n_epochs}\nKL Divergence: {kl_div:.4f}\nMin Loss: {min_loss:.4f} at Epoch {min_loss_epoch}"
plt.text(0.6, 0.95, text_str, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.7))

# Save the training loss plot
loss_name = f"loss.png"
loss_plot_path = os.path.join(args.outdir, loss_name)
plt.savefig(loss_plot_path)

# Function to plot marginal distributions
def plot_marginals(target, trained, prior, feature_names, outdir):
    num_features = target.shape[1]
    
    for i in range(num_features):
        plt.figure(figsize=(8, 6))
        
        # Plot target distribution
        plt.hist(target[:, i], bins=50, alpha=0.5, label='Target', density=True, color='blue')
        
        # Plot trained distribution
        plt.hist(trained[:, i], bins=50, alpha=0.5, label='Trained', density=True, color='green')
        
        # Plot prior distribution
        plt.hist(prior[:, i], bins=50, alpha=0.5, label='Prior', density=True, color='gray')
        
        # Add labels and title
        plt.xlabel(feature_names[i])
        plt.ylabel("Density")
        plt.title(f"Marginal Distribution for {feature_names[i]}")
        plt.legend()
        
        # Save the plot
        plot_path = os.path.join(outdir, f"marginal_feature_{i+1}.png")
        plt.savefig(plot_path)
        plt.close()

# Call the function with the necessary arguments
feature_names = ["b-tagging score", "Energy scaled"]
plot_marginals(bkg_coord_scaled[:10000], trained, prior, feature_names, args.outdir)


