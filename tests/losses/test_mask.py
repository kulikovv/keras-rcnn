import numpy as np
import keras.backend
from keras_rcnn.losses.mask import _mask_loss

def test_maskloss():
    eps = 1e-6
    a = keras.backend.variable(np.zeros((10,50,50)))
    b = keras.backend.variable(np.ones((10, 50, 50)))
    assert (keras.backend.eval(_mask_loss(a,a)) < eps)
    assert (keras.backend.eval(_mask_loss(b,b)) < eps)
    assert ((keras.backend.eval(_mask_loss(a, b))) > 10.)