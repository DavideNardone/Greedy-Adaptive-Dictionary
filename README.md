# Greedy-Adaptive-Dictionary

This is a python implementation of [(Jafari et. al)](http://ieeexplore.ieee.org/document/5776648/). They present a greedy adaptive dictionary learning algorithm that sets out to find sparse atoms for speech signals. The algorithm learns the dictionary atoms on data frames taken from a speech signal. It iteratively extracts the data frame with minimum sparsity index, and adds this to the dictionary matrix. The contribution of this atom to the data frames is then removed, and the process is repeated. The algorithm is found to yield a sparse signal decomposition, supporting the hypothesis of a link between sparsity in the decomposition and dictionary. The algorithm is applied to the problem of speech representation and speech denoising, and its performance is compared to other existing methods. 


# Prerequisites

  - Python 2.7 or greater <br>
  - Librosa or any other package for reading audio signals
  
# Usage

`git clone https://github.com/DavideNardone/Greedy-Adaptive-Dictionary.git` <br>

`unzip Greedy-Adaptive-Dictionary-master.py`

then... run the following python file:

`GAD.py (naive example)` <br>

# Authors

Davide Nardone, University of Naples Parthenope, Science and Techonlogies Departement,<br> Msc Applied Computer Science <br/>
https://www.linkedin.com/in/davide-nardone-127428102

# Contacts

For any kind of problem, questions, ideas or suggestions, please don't esitate to contact me at: 
- **davide.nardone@studenti.uniparthenope.it**

# References

[Jafari et. al]: Jafari, Maria G., and Mark D. Plumbley. "Fast dictionary learning for sparse representations of speech signals." IEEE Journal of Selected Topics in Signal Processing 5.5 (2011): 1025-1031.
