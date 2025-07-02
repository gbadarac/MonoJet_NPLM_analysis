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

## to run on slurm cluster:
- `submit_toy.sh`: submits the script `toy.py` with arguments (example: `python toy.py -m h_4 -s 0 -b 10000 -r 50000 -t 100 -l 5`)
- `toy.py`: runs multiple toys according to the arguments
  -   `-m`, `--manifold`, type=str, help="manifold type (must be in folders.keys())", required=True
  -   `-s`, `--signal`, type=int, help="signal (number of signal events)", required=True
  -   `-b`, `--background`, type=int, help="background (number of background events)", required=True
  -   `-r`, `--reference`, type=int, help="reference (number of reference events, must be larger than background)", required=True
  -   `-t`, `--toys`, type=int, help="toys", required=True)
  -   `-l`, `--signalclass`, type=int, help="class number identifying the signal", required=True

To run:
  ```
  sbatch submit_toy.sh
  ```
