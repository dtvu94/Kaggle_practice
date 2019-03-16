# Integrated Gradients
![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)  
This is the chainer implementation of ["Axiomatic Attribution for Deep Networks"](https://arxiv.org/pdf/1703.01365.pdf). The original tensorflow version could be found [here](https://github.com/ankurtaly/Integrated-Gradients).
## Acknowledgement
- [ankurtaly's tensorflow version](https://github.com/ankurtaly/Integrated-Gradients)
## Requirements
- python-3.6
- chainer-5.3.0
- opencv-python
## Instructions

### The library support all networks which are implemented in Chainer (of course, you can add any networks by yourself)
### Run the code
```bash
python main.py --model= vgg16  --image= 01.jpg

```
## Results
Results are slightly different from the original paper.
Reason:
- Different normalization
- Different framework