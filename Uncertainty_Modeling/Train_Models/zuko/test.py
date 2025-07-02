import zuko
model = zuko.flows.NSF(features=2, transforms=3, hidden_features=[32, 32], bins=8, bayesian=True)
print(model)
