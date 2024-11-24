import os
import tempfile
BACKEND = 'jax'
os.environ['KERAS_BACKEND'] = BACKEND

import pytest
import keras
from keras import ops
from keras import backend
from keras import random
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from sigkan import SigDense, SigKAN

def generate_random_tensor(shape):
    return random.normal(shape=shape, dtype=backend.floatx())

@pytest.fixture(params=[SigDense, SigKAN])  # Test different signature depths
def SignatureWeightedLayer(request):
    """Fixture providing different signature depths for testing.
    
    Parametrizes tests with signature depths of 2, 3, and 4 to ensure 
    the signature computation works correctly at different truncation levels.
    
    Args:
        request: pytest request object containing the parameter value
    
    Returns:
        int: Signature computation depth (2, 3, or 4)
    """
    return request.param

@pytest.fixture(params=[1, 2, 3])  # Test different signature depths
def sig_level(request):
    """Fixture providing different signature depths for testing.
    
    Parametrizes tests with signature depths of 2, 3, and 4 to ensure 
    the signature computation works correctly at different truncation levels.
    
    Args:
        request: pytest request object containing the parameter value
    
    Returns:
        int: Signature computation depth (2, 3, or 4)
    """
    return request.param

def test_fit_save_and_load(SignatureWeightedLayer, sig_level):
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 10, 8
    units = 16

    # Create and compile the model
    inputs = Input(shape=(time_steps, features))
    x = Dense(5)(inputs)
    sig_layer = SignatureWeightedLayer(units, sig_level=2)
    x = sig_layer(x)
    x = Flatten()(x)
    outputs = Dense(units)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    # Generate some random data
    x_train = generate_random_tensor((batch_size, time_steps, features))
    y_train = generate_random_tensor((batch_size, units))

    # Train the model
    model.fit(x_train, y_train, epochs=1, batch_size=16, verbose=False)

    # Get predictions before saving
    predictions_before = model.predict(x_train, verbose=False)

    # Save the model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'sig_model.keras')
        model.save(model_path)

        # Load the model
        loaded_model = load_model(model_path)

    # Get predictions after loading
    predictions_after = loaded_model.predict(x_train, verbose=False)

    # Compare predictions
    assert ops.all(ops.equal(predictions_before, predictions_after)), "Predictions should be the same after loading"

    # Test that the loaded model can be used for further training
    loaded_model.fit(x_train, y_train, epochs=1, batch_size=16, verbose=False)

    print("SigWeighted model successfully saved, loaded, and reused.")
