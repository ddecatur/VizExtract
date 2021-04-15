import cv2
import tensorflow as tf

def prepare(filepath):
    """A function to prepare an image to be processed by the model
    
    Input:
    ------
    filepath: str, a filepath to the image to be processed

    Return:
    -------
    new_array: an image array of the prepared image
    """

    IMG_SIZE = 150
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

def load_model(fname):
    """A function to load a model with Keras given
        a path to that model

    """

    return tf.keras.models.load_model(fname)

def predict(img, model):
    """A function to obtain a prediction on the given image with the given model

    Parameters:
    -----------
    img: an image array
    model: a Keras model object

    Returns:
    --------
    prediction: np.array(shape=(2,2)), a 2-D array but the only the first entry
        in the outermost dimension is relevent. This first element is an array
        of floats corresponding to the predicted likelyhood of that respective
        index being the category of the image
    """

    prediction = model.predict([prepare(img)])
    return prediction

def predictCategory(img,model,CATEGORIES=[1,2,3]):
    """A function to take the prediction from "predict" and return the
        associated category from the list provided in CATEGORIES
    
    Parameters:
    -----------
    img: an image array
    model: a Keras model object
    CATEGORIES: list, a list of potential classes

    Returns:
    --------
    CATEGORIES[midx]: str, a string corresponding to the predicted class
        (the index into the the CATEGORIES list correspoinding to the index into
        "prediction" that has the maximum value)
    """

    prediction = predict(img, load_model(model))
    midx = 0
    mval = prediction[0][0]
    for i,pred in enumerate(prediction[0]):
        if pred > mval:
            midx = i
            mval = pred
    
    return CATEGORIES[midx]
