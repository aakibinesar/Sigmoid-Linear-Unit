# IMPORTS
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.noise import AlphaDropout
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.optimizers import Adam

# LOAD DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# PREPROCESSING
def preprocess_mnist(x_train, y_train, x_test, y_test):
    # Normalizing all images of 28x28 pixels
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    
    # Float values for division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # Normalizing the RGB codes by dividing it to the max RGB value
    x_train /= 255
    x_test /= 255
    
    # Categorical y values
    y_train = to_categorical(y_train)
    y_test= to_categorical(y_test)
    
    return x_train, y_train, x_test, y_test, input_shape
    
x_train, y_train, x_test, y_test, input_shape = preprocess_mnist(x_train, y_train, x_test, y_test)

# Readying neural network model
def build_dnn(activation,
              dropout_rate,
              optimizer):
    model = Sequential()
    
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation=activation, kernel_initializer="normal"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation=activation, kernel_initializer="normal"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation=activation, kernel_initializer="normal"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation=activation, kernel_initializer="normal"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation=activation, kernel_initializer="normal"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation=activation, kernel_initializer="normal"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation=activation, kernel_initializer="normal"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation=activation, kernel_initializer="normal"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=optimizer, 
        metrics=['accuracy']
    )
    
    return model

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
get_custom_objects().update({'gelu': Activation(gelu)})

def silu(x):
    return x * (1 / (1 + (tf.math.exp(-x))))
get_custom_objects().update({'silu': Activation(silu)})

act_func = ['relu','elu','gelu', 'silu']

result = []

for activation in act_func:
    print('\nTraining with -->{0}<-- activation function\n'.format(activation))
    
    model = build_dnn(activation=activation,
                      dropout_rate=0,
                      optimizer=Adam())
    
    history = model.fit(x_train, y_train,
          validation_split=0.20,
          batch_size=128, # 128 is faster, but less accurate. 16/32 recommended
          epochs=10,
          verbose=1,
          validation_data=(x_test, y_test))
    
    result.append(history)
    
    K.clear_session()
    del model

print(result)

new_act_arr = act_func[:]
new_results = result[:]

def plot_act_func_results(results, activation_functions = []):
    plt.figure(figsize=(10,10))
    plt.style.use('default')
    
    # Plot validation accuracy values
    for act_func in results:
        plt.plot(act_func.history['val_accuracy'])
        
    plt.title('Model accuracy')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Epoch')
    plt.legend(activation_functions)
    plt.show()

    # Plot validation loss values
    plt.figure(figsize=(10,10))
    
    for act_func in results:
        plt.plot(act_func.history['val_loss'])
        
    plt.title('Model loss')
    plt.ylabel('Test Loss')
    plt.xlabel('Epoch')
    plt.legend(activation_functions)
    plt.show()

plot_act_func_results(new_results, new_act_arr)
