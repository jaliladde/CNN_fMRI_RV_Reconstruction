from keras import backend as K


def R2(y_true, y_pred):
    res =  K.sum(K.square( y_true-y_pred ))
    tot = K.sum(K.square(y_true-K.mean(y_true)))
    return (1-res/(tot+K.epsilon()))
