import keras
from keras.layers import Layer, Dense, Dropout

from keras_efficient_kan import KANLinear
from keras_sig import SigLayer

from sigkan import GRKAN, GRN

class SigKAN(Layer):
    def __init__(self, unit, sig_level, dropout = 0., **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.sig_level = sig_level
        self.sig_layer = SigLayer(self.sig_level)
        self.kan_layer = KANLinear(unit, dropout = dropout, use_bias = False, use_layernorm = False)
        self.sig_to_weight = GRKAN(unit, activation = 'softmax', dropout = dropout)
        self.dropout = Dropout(dropout)

    def build(self, input_shape):
        _, seq_length, n_features = input_shape
        name = self.name
        self.time_weigthing_kernel = self.add_weight(
            shape=(seq_length, 1),
            name=f"{name}_time_weigthing_kernel",
        )
        super().build(input_shape)
        
    def call(self, inputs):
        inputs = self.time_weigthing_kernel * inputs
        sig = self.sig_layer(inputs)
        weights = self.sig_to_weight(sig)
        kan_out = self.kan_layer(inputs)
        kan_out = self.dropout(kan_out)
        return kan_out * keras.ops.expand_dims(weights, axis=1)

class SigDense(Layer):
    def __init__(self, unit, sig_level, dropout = 0., **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.sig_level = sig_level
        self.sig_layer = SigLayer(self.sig_level)
        self.dense_layer = Dense(unit)
        self.sig_to_weight = GRN(unit, activation = 'softmax', dropout = dropout)
        self.dropout = Dropout(dropout)

    def build(self, input_shape):
        _, seq_length, n_features = input_shape
        name = self.name
        self.time_weigthing_kernel = self.add_weight(
            shape=(seq_length, 1),
            name=f"{name}_time_weigthing_kernel",
        )
        super().build(input_shape)
        
    def call(self, inputs):
        inputs = self.time_weigthing_kernel * inputs
        sig = self.sig_layer(inputs)
        weights = self.sig_to_weight(sig)
        dense_out = self.dense_layer(inputs)
        dense_out = self.dropout(dense_out)
        return dense_out * keras.ops.expand_dims(weights, axis=1)

