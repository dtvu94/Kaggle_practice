import keras
from keras.layers import Input, Dense, Flatten, Dense, Dropout
from keras.layers import Conv1D, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model, Sequential
from keras.optimizers import SGD

def Lenet5():
    img_shape = (32, 32, 1)
    Img_Input = Input(shape=img_shape)
    
    x = Conv2D(filters=6, kernel_size=5, strides=1, activation='relu')(Img_Input)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Conv2D(filters=16, kernel_size=5, strides=1, activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(units=120, activation='relu')(x)
    x = Dense(units=84, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)
    
    model = Model(inputs=Img_Input, outputs=x)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model

model = Lenet5()