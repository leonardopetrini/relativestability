# Compute stabilities of trained netowrks

Compute deformation (D_k), noise (G_k) and relative (R_k) stabilities for trained networks.

Run `main.py`. Args:

- `net`, network architecture
- `dataset`, input dataset
- `init`, if `1`, compute stabilities at initialization
- `by_layer`, if `1`, compute stabilities at layer by layer
- `P`, number of testing points to use for the computations
- `gaussian_corruption_std`, std of Gaussian corruption added to testing points 
- `corrupt_test`, if `1`, corrupt test set also when computing accuracy
- `shuffle_channels`, shuffle channels when computing stabilities

See `experiments_ICLR23.md` for examples on how to run the code.