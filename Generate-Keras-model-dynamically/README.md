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

### Result:
Depend on the setting, the file will be saved to a location.
I put 3 sample json files with 3 result files from the progress.

## Syntax design in the json file
Have 8 parts:
- Info
- Parameters
- Variables
- Layer definitions
- Layer connections
- Model
- Compile
- Additional functions

The detail of each part is below, please suggest an idea if you want to update to perfect its syntax.
Each part is saved as the dictionary format in python.

Note: This module allows **ASCII** characters only. In case of unicode, you have to fix the IO part by yourself.

### Info
This part contains information for creating files with many status, conditions. The list below describes each key and its role in the part.

1. File_path: an absolute path of the file point to a location where it is saved to.
    For examples:
    - E:\GitProjects\Machine_Learning_practice\Generate-Keras-model-dynamically\one.py
    - E:\GitProjects\Machine_Learning_practice\Generate-Keras-model-dynamically\two.py
2. Write_mode: can be append or write mode, intend to combine with existed sources.
    For examples: "a", "w"
3. Keras_mode: sequential or function type in creating model
4. Function_name: the name of the function which create the required model
    For examples:
    - Function name: "MyModel" 
    => result: def MyModel():
5. Comment_mode: a flag to turn on/off writing comments to the creating model function
    For examples: "True"/"Yes", "False"/"No"

### Parameters


### Variables


### Layer definitions


### Layer connections


### Model


### Compile


### Additional functions


