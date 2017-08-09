import keras.backend
import keras.losses

def _mask_loss(y_true,y_pred):
    """
    Mask loss is just a simple binary cross entropy. In keras logits must be flatten.
    :param y_true: Ground truth 
    :param y_pred: Predicted logits
    :return: $\frac{1}{N}-\sum(ylog(yp+eps)+(1-y)log(1-yp+eps))$
    """
    return keras.backend.mean(keras.losses.binary_crossentropy(keras.backend.batch_flatten(y_true),keras.backend.batch_flatten(y_pred)))
