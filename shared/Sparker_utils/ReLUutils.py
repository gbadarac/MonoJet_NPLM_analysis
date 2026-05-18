import torch
import torch.nn as nn

class ShallowLinear(nn.Module):
    def __init__(self, architecture=[2, 3, 1], weights_clip=None):
        super(ShallowLinear, self).__init__()
        self.n_layers = len(architecture)-1
        self.layers = nn.ModuleList([torch.nn.Linear(architecture[i], architecture[i+1], bias=True) for i in range(self.n_layers)])
        self.relu = torch.nn.LeakyReLU()
        self.weights_clip = weights_clip
        
    def call(self, x):
        out = x
        for i in range(self.n_layers-1):
            out = self.relu(self.layers[i](out))
        out = self.layers[-1](out)
        return out 

    def clip(self):
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                w = layer.weight.data
                w = w.clamp(-1*self.weights_clip, self.weights_clip)
                layer.weight.data = w

            if hasattr(layer, 'bias'):
                w = layer.bias.data
                w = w.clamp(-1*self.weights_clip, self.weights_clip)
                layer.bias.data = w
        return
        
    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.weight.data)
        return weights

    def get_biases(self):
        biases = []
        for layer in self.layers:
            biases.append(layer.bias.data)
        return biases

    def L1regularizer(self):
        return sum(p.abs().sum() for p in self.parameters())

    def L2regularizer(self):
        return sum(p.pow(2).sum() for p in self.parameters())