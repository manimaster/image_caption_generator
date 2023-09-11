# Functions to save and load models, weights, or any other utility functions.
def save_model(model, filepath):
    model.save(filepath)

def load_model(filepath):
    return keras.models.load_model(filepath)
