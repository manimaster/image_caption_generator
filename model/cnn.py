from keras.applications.inception_v3 import InceptionV3
from keras.models import Model

def cnn_model(input_shape):
    model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    new_input = model.input
    hidden_layer = model.layers[-1].output

    model_new = Model(inputs=new_input, outputs=hidden_layer)
    return model_new
