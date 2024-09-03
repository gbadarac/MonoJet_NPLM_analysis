# NPLM-embedding
code to apply the NPLM method to embeddings

Reference papers:
- *“Learning New Physics from a Machine”* [https://doi.org/10.1103/PhysRevD.99.015014] :  here you can find the main idea behind NPLM.
- *“Learning New Physics Efficiently with non-parametric models”* [https://link.springer.com/article/10.1140/epjc/s10052-022-10830-y]:  here you can find the technical details about the kernel methods implementation, in particular how to choose the hyper-parameters M (number of centroids), $\sigma$ (kernels' width) and $\lambda$ (L2 regularization strenght).
- *“Multiple testing for signal-agnostic searches of new physics with machine learning”* [https://arxiv.org/abs/2408.12296]: here you can find a way to combine multiple values of $\sigma$ into a single hyper-test.

## instruction to install the Falkon library
Follow the instructions in [https://falkonml.github.io/falkon/install.html]

## simple piece of code to start off:
- `run-NPLM-toy.ipynb`
- `run-NPLM-toy.py`
