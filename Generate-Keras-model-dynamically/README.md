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
python main.py --input-config {name/relative path of json config file}
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
- Write_mode: can be append or write mode, intend to combine with existed sources. 
    - For examples: "a", "w"
- Function_name: the name of the function which create the required model.
    - For examples: Function name: "MyModel" => result: def MyModel():

### Parameters
Information:
- Contains all input parameters for the model function and their descriptions.
Syntax:
- All keys' characters must be in CAPITAL letter.
    - For examples: "MIN_X", "KERNEL_SIZE_1", "POOL_2_2", ...
- All values must be in string type.
    - For examples: "(2, 2)", "3", "Conv1D", ...

### Variables
Information:
- Contains all local variables for the model function and their descriptions.
Syntax:
- All keys' characters must be in CAPITAL letter
    - For examples: "MIN_X", "KERNEL_SIZE_1", "POOL_2_2", ...
- All values must be in string type
    - For examples: "(2, 2)", "3", "Conv1D", ...

### Layer definitions
Information:
- Contains layers' definitions such as Convolution, flatten, ...
- Key is the layer exact name.
- Value is the layer input parameters
Syntax:
- Layer's key can be anything, depend on your hobbies;
    - For examples: "1", "2", "conv1", "conv2", "conv3", "conv4", ...
- The attribute key: "Layer" must be written correctly.
- Other attributes' keys of one layer must be written the same as the API in Keras.
    - For intance: "filters" key of layer "Conv1D"
- The template: {VARIABLE_NAME} is used for assigning a value in "variables" or "Additional functions" part to the attribute value of this part.
    - For examples: "Layer": "Con1", "kernel_size": "SIZE_3", ... 

### Layer connections
Information:
- The place includes all connections between layers to form a network
Syntax:
- Template for connection: {NAME_CONNECT} = ARRAY OF {LAYER_DEFINITION_KEY_NAME} or {PREVIOUS_DEFINE_CONNECTION}
- The special case for the input: {NAME_CONNECT} = ARRAY OF {INPUT_LAYER_KEY_NAME}

### Model
Information:
- Contains all input parameters for the model constructor function.
Syntax:
- "inputs" attribute value can be a {LAYER_DEFINITION_KEY_NAME} or an array of {LAYER_DEFINITION_KEY_NAME}
- "outputs" attribute value can be a {LAYER_DEFINITION_KEY_NAME} or an array of {LAYER_DEFINITION_KEY_NAME}

### Compile
Information:
- Contains all input parameters for the model compile function.
Syntax:
- Attributes' keys of one layer must be written the same as the API in Keras.
- Attributes's values can be anything as it fits the syntax of the create function
    - For examples: "loss": "my_loss_func", "optimizer": "my_opt_func", ...

### Additional functions
Information:
- Implement additional functions in the model file. 
- Key of an element is a symbol name to search if it is used in Layers_structure/Model/Compile
- Value of an element is the content of the function which is created by a convert function.
Syntax:
- All keys' characters must be in CAPITAL letter
- All values must be absolutely the same as the content from the function: ```convert_py_func_to_txt```
