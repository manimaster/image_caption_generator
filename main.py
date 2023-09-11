from model import cnn, rnn
from data import preprocessing
from utils import load_save

def train_model():
    # Load data
    images, captions = load_data()

    # Preprocess data
    image_features, tokenized_captions = preprocessing.preprocess_data(images, captions)

    # Build the models
    cnn_model = cnn.cnn_model((299, 299, 3))
    rnn_model = rnn.rnn_model(vocab_size, max_length)

    # Combine the models and train
    # ...

def generate_caption(image):
    # Load models
    cnn_model = load_save.load_model('pretrained/cnn_model.h5')
    rnn_model = load_save.load_model('pretrained/rnn_model.h5')

    # Extract features
    features = cnn_model.predict(image)

    # Generate caption using rnn_model
    # ...

    return caption
