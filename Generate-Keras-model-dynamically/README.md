# Generate Keras model dynamically

## Requirements
- python-3.6
- tensorflow
- keras
- numpy

## The project is a solution for an issue:
- Need to have a module to make an architecture of a Keras model.
- Want to create a Keras model dynamically at runtime
- Have a sample of loading Keras model at runtime
- The structure of Keras model file is flexible and easy to use

## Instructions

### Run the code
Template:
```bash
python main.py --input-config [name/relative path of json config file]
```
Example:
```bash
python main.py --input-config lenet5.json
```

### Syntax design in the json file
Have 8 parts:
- Info
- Parameters
- Variables
- Layer definitions
- Layer connections
- Model
- Compile
- Additional functions

The detail of each part is below, please suggest an idea if you want to update to perfect its syntax

**Info**


**Parameters**


**Variables**


**Layer definitions**


**Layer connections**


**Model**


**Compile**



**Additional functions**
