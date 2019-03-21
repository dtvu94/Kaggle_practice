import keras
from keras.layers import Input, Dense, Flatten, Dense, Dropout
from keras.layers import Conv1D, Conv2D, MaxPooling2D, ZeroPadding2D, concatenate
from keras.models import Model, Sequential
from keras.optimizers import SGD

def Inception_module():
    Img_Input = Input(shape=(256, 256, 3))

    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(Img_Input)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(Img_Input)
    tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(Img_Input)
    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

    output = concatenate([tower_1, tower_2, tower_3], axis=1)
    
    model = Model(inputs=Img_Input, outputs=output)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model

model = Inception_module()