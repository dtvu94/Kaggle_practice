# Integrated Gradients
![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)  
This is the chainer implementation of ["Axiomatic Attribution for Deep Networks"](https://arxiv.org/pdf/1703.01365.pdf). The original tensorflow version could be found [here](https://github.com/ankurtaly/Integrated-Gradients).
The source function to calculate the gradient is complete but the visualization part is not. I use the visualization library from the orginal one but it seems not good.

## Acknowledgement
- [ankurtaly's tensorflow version](https://github.com/ankurtaly/Integrated-Gradients)

## Requirements
- python-3.6
- chainer-5.3.0
- opencv-python
- imageio

## The project is a solution for an issue:
- Want to use Integrated Gradients method in Chainer framework
- Cannot find a single project which satisfy this

## Instructions

### The library support networks which are implemented in Chainer:
(of course, you can add any networks by yourself)
1. VGG16Layers - named: vgg16
2. VGG19Layers - named: vgg19
3. GoogLeNet - named: googlenet
4. ResNet50Layers - named: resnet50
5. ResNet101Layers - named: resnet101
6. ResNet152Layers - named: resnet152

### Run the code
Template:
```bash
python main.py --model [model_name]  --image= [image_name]
```
Example:
```bash
python main.py --model vgg16 --image 01.jpg
```
## Results
Results are slightly different from the original paper.
Reason:
- Different normalization
- Different framework