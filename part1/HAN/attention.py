import tensorflow.keras as keras
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from tensorflow.keras import backend as K


class Attention(Layer):
    def __init__(self, attention_dim):
        self.dim = attention_dim
        self.init = initializers.get('glorot_uniform')
        super(Attention, self).__init__()
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(
            name="W",
            shape=(int(input_shape[-1]), dim),
            initializer=self.init,
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.attention_dim,),
            initializer='zero',
            trainable=True,
        )
        self.u = self.add_weight(
            name="u",
            shape=(self.attention_dim, 1),
            initializer=self.init,
            trainable=True,
        )
        super(Attention, self).build(input_shape)
    
    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, axis=-1)
        ait = K.exp(ait)
        if mask is not None:
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


