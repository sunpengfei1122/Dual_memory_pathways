# Algorithm-hardware co-design of neuromorphic networks with dual memory pathways

Pengfei Sun* [1], Zhe Su*[2], Jascha Achterberg* [3], Giacomo Indiveri [2] , Dan F.M. Goodman [1], Danyal Akarca [1, 4]

Imperial College London, ETH Zurich, University of Oxford, University of Cambridge

\* These authors contributed equally to this work.

Corresponding: Danyal Akarca
# README #
## Requirements
Python 3 with the following packages installed:

* PyTorch 
* numpy
* spikingjelly

The software has been tested with CUDA libraries version 11.3 and Pytorch 1.12.1, and spikingjelly==0.0.0.0.14. 

`pip install torch==1.12.1+cu113  torchvision==0.13.1+cu113  torchaudio==0.12.1    -f https://download.pytorch.org/whl/cu113/torch_stable.html`

`pip install spikingjelly`

## Examples
Example implementations can be found inside different dataset folders. For the SHD/SSC, please download the dataset from  (https://zenkelab.org/datasets/).

* Run example SHD implementation, 

```bash
   cd shd/src

   python train_spiking.py -d 5 -t 40  
```
d is the memory state and t is the state buffer length.
	
