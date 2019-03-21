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
Information:
- Store information of the creating process

Syntax:
- File_path: an absolute path of the creating Keras model file.
- Write_mode: can be append or write mode, intend to combine with 
existed sources. 
    - For examples: "a", "w"
- Function_name: the name of the function which create the 
required model.
    - For examples: Function name: "MyModel" => result: def MyModel():
- Comment_mode: a flag to turn on/off writing comments to the 
creating model function.
    - For examples: "True"/"Yes", "False"/"No"

### Parameters
Information:
    Contains all input parameters for the model function and their
    descriptions.
Syntax:
    a)  All keys' characters must be in CAPITAL letter.
        For examples: "MIN_X", "KERNEL_SIZE_1", "POOL_2_2", ...
    b)  All values must be in string type.
        For examples: "(2, 2)", "3", "Conv1D", ...

### Variables
Information:
    Contains all local variables for the model function and their
    descriptions.
Syntax:
    a)  All keys' characters must be in CAPITAL letter
        For examples: "MIN_X", "KERNEL_SIZE_1", "POOL_2_2", ...
    b)  All values must be in string type
        For examples: "(2, 2)", "3", "Conv1D", ...

### Layer definitions
Information:
    Contains layers' definitions such as Convolution, flatten, ...
    Key is the layer exact name.
    Value is the layer input parameters
Syntax:
    a)  Layer's key can be anything, depend on your hobbies;
        For examples: "1", "2", "conv1", "conv2", "conv3", "conv4", ...
    b)  The attribute key: "Layer" must be written correctly.
    c)  Other attributes' keys of one layer must be written the same as 
    the API in Keras.
        For intance: "filters" key of layer "Conv1D"
    d)  The template: [VARIABLE_NAME] is used for assigning a value in
    "variables" part to the attribute value of this part.
        For examples: "Layer": "Con1", "kernel_size": "SIZE_3", ... 

### Layer connections
Information:
    The place includes all connections between layers to form a network
Syntax:
    a)  Template for connection:
        [NAME_CONNECT] = ARRAY OF [LAYER_DEFINITION_KEY_NAME] 
                                or [PREVIOUS_DEFINE_CONNECTION]
    b)  The special case for the input:
            [NAME_CONNECT] = ARRAY OF [INPUT_LAYER_KEY_NAME]

### Model
Information:
    Contains all input parameters for the model constructor function.
Syntax:
    a)  "inputs" attribute value can be a [LAYER_DEFINITION_KEY_NAME] or 
    an array of [LAYER_DEFINITION_KEY_NAME]
    b)  "outputs" attribute value can be a [LAYER_DEFINITION_KEY_NAME] 
    or an array of [LAYER_DEFINITION_KEY_NAME]

### Compile
Information:
    Contains all input parameters for the model compile function.
Syntax:
    a)  Attributes' keys of one layer must be written the same as the
    API in Keras.
    b)  Attributes's values can has the template for functions in 
    "functions_definition" part
        For examples: 
            "loss": "[VAE_LOSS]"
            "optimizer": "[MYOPTIMIZE]", ...

### Additional functions
Information:
    Implement additional functions in the model file. 
    Key of an element is a symbol name to search if it is used in 
        Layers_structure/Model/Compile
    Value of an element is the content of the function which is created
        by a convert function.
Syntax:
    a)  All keys' characters must be in CAPITAL letter
    b)  All values must be absolutely the same as the content from the
    function:
        convert_py_func_to_txt

