import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Add, LayerNormalization, Multiply, Dropout
from tkan import KANLinear

class AddAndNorm(Layer):
    def __init__(self, **kwargs):
        super(AddAndNorm, self).__init__(**kwargs)
        self.add_layer = Add()
        self.norm_layer = LayerNormalization()
    
    def call(self, inputs):
        tmp = self.add_layer(inputs)
        tmp = self.norm_layer(tmp)
        return tmp

    def compute_output_shape(self, input_shape):
        return input_shape[0]  # Assuming all input shapes are the same

class Gate(Layer):
    def __init__(self, hidden_layer_size = None, **kwargs):
        super(Gate, self).__init__(**kwargs)
        self.hidden_layer_size = hidden_layer_size

    def build(self, input_shape):
        if self.hidden_layer_size is None:
            self.hidden_layer_size = input_shape[-1]
        self.dense_layer = Dense(self.hidden_layer_size)
        self.gated_layer = Dense(self.hidden_layer_size, activation='sigmoid')
        super(Gate, self).build(input_shape)

    def call(self, inputs):
        dense_output = self.dense_layer(inputs)
        gated_output = self.gated_layer(inputs)
        return Multiply()([dense_output, gated_output])

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.hidden_layer_size,)


class GRKAN(Layer):
    def __init__(self, hidden_layer_size, output_size=None, activation = None, dropout = 0.1, **kwargs):
        super(GRKAN, self).__init__(**kwargs)
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.activation = tf.keras.activations.get(activation) if activation is not None else None
        self.dropout = Dropout(dropout)
        self.dropout_value = dropout

    def build(self, input_shape):
        if self.output_size is None:
            self.output_size = self.hidden_layer_size
        self.skip_layer = Dense(self.output_size)
        
        self.hidden_layer_1 = KANLinear(self.hidden_layer_size, base_activation='elu', dropout = self.dropout_value)
        self.hidden_layer_2 = KANLinear(self.hidden_layer_size, dropout = self.dropout_value)
        self.gate_layer = Gate(self.output_size)
        self.add_and_norm_layer = AddAndNorm()
        super(GRKAN, self).build(input_shape)

    def call(self, inputs):
        if self.skip_layer is None:
            skip = inputs
        else:
            skip = self.skip_layer(inputs)
        
        hidden = self.hidden_layer_1(inputs)
        hidden = self.hidden_layer_2(hidden)
        hidden = self.dropout(hidden)
        gating_output = self.gate_layer(hidden)
        output = self.add_and_norm_layer([skip, gating_output])
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_size,)


class GRN(Layer):
    def __init__(self, hidden_layer_size, output_size=None, activation = None, dropout = 0.1, **kwargs):
        super(GRN, self).__init__(**kwargs)
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.activation = tf.keras.activations.get(activation) if activation is not None else None
        self.dropout = Dropout(dropout)
        self.dropout_value = dropout

    def build(self, input_shape):
        if self.output_size is None:
            self.output_size = self.hidden_layer_size
        self.skip_layer = Dense(self.output_size)
        
        self.hidden_layer_1 = Dense(self.hidden_layer_size, activation='elu')
        self.hidden_layer_2 = Dense(self.hidden_layer_size)
        self.gate_layer = Gate(self.output_size)
        self.add_and_norm_layer = AddAndNorm()
        super(GRN, self).build(input_shape)

    def call(self, inputs):
        if self.skip_layer is None:
            skip = inputs
        else:
            skip = self.skip_layer(inputs)
        
        hidden = self.hidden_layer_1(inputs)
        hidden = self.hidden_layer_2(hidden)
        hidden = self.dropout(hidden)
        gating_output = self.gate_layer(hidden)
        output = self.add_and_norm_layer([skip, gating_output])
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_size,)