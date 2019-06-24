
# Isospectralization
Implementation of the paper: Cosmo et al., "Isospectralization, or how to hear shape, style, and correspondence", arXiv 2018 https://arxiv.org/abs/1811.11465

The paper describes a method to optimize the R^2/R^3 embedding of a discretized surface (triangular mesh) as to align the eigenvalues of its LBO to a target set of eigenvalues (possibly derived from a target shape).

The code is sub-divided in two folders containing the code for the relative embedding space (R^2 and R^3).
Each folder contains the following python files:
- test.py - small script to load the data and run the optimization
- shape_library.py - a library containing functions to load and pre-process the shape
- spectrum_alignment.py -  tensorflow code performing the optimization

If you are using this code, please cite:

```
@InProceedings{Cosmo_2019_CVPR,
  author = {Cosmo, Luca and Panine, Mikhail and Rampini, Arianna and Ovsjanikov, Maks and Bronstein, Michael M. and Rodola, Emanuele},
  title = {Isospectralization, or How to Hear Shape, Style, and Correspondence},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```
