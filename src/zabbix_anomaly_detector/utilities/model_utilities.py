import numpy as np
from tensorflow.keras.models import load_model as keras_load_model

# Internal variable to hold the model instance
_model = None
_model_path = "./model/discriminator_saved_model.keras"

def get_model():
    """
    Lazily loads and caches the model on first use.
    """
    global _model
    if _model is None:
        _model = keras_load_model(_model_path)
        _model.trainable = False
    return _model

def get_anomaly_scores(X: np.ndarray) -> np.ndarray:
    model = get_model()
    preds = model.predict(X, verbose=0)
    return preds
