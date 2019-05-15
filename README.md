
# Isospectralization
Implementation of the paper: Cosmo et al., "Isospectralization, or how to hear shape, style, and correspondence", arXiv 2018 https://arxiv.org/abs/1811.11465

The paper describes a method to optimize the R^2/R^3 embedding of a discretized surface (triangular mesh) as to align the eigenvalues of the spectral decomposition of its LBO to a target set of eigenvalues (possibly derived from a target shape).

The code is sub-divided in two folders containing the code for the relative embedding space (R^2 and R^3).
Each folder contains the following python files:
- test.py - small script to load the data and run the optimization
- shape_library.py - a library containing functions to load and pre-process the shape
- spectrum_alignment.py -  tensorflow code performing the optimization

If you are using this code, please cite:

@article{DBLP:journals/corr/abs-1811-11465,
  author    = {Luca Cosmo and
               Mikhail Panine and
               Arianna Rampini and
               Maks Ovsjanikov and
               Michael M. Bronstein and
               Emanuele Rodol{\`{a}}},
  title     = {Isospectralization, or how to hear shape, style, and correspondence},
  journal   = {CoRR},
  volume    = {abs/1811.11465},
  year      = {2018},
  url       = {http://arxiv.org/abs/1811.11465},
  archivePrefix = {arXiv},
  eprint    = {1811.11465},
  timestamp = {Fri, 30 Nov 2018 12:44:28 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1811-11465},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
