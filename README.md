# SigKAN: Signature-Weighted Kolmogorov-Arnold Networks for Time Series

![SIGKAN representation](image/SigKAN.drawio.png)

This folder includes the original code implemented for the [paper of the same name](https://arxiv.org/abs/2406.17890).

A pypi package is available at [pypi](https://pypi.org/project/sigkan/)

The SigKAN is a novel layer that combines the power of path signature and Kolmogorov-Arnold Networks.

The idea behing is to use a learnable path signature that is transformed in weights to the KAN layer.

The Signature is passed through a GRKAN (Gated Residual KAN unit) that is a modified GRN where some Dense layers are replaced by KAN layers.

New in version 0.2.0:

The signature is now computed using [keras_sig](https://github.com/remigenet/keras_sig) instead of [iisignature_tensorflow_2](https://github.com/remigenet/iisignature-tensorflow-2/) that is a keras implementation of the signature based on [signatory](https://github.com/patrick-kidger/signatory) and [signax](https://github.com/anh-tong/signax/). Keras_sig being in pure keras3 the package is compatible and tested with all keras backend (tensorflow2, jax and torch). However, we recommend strongly to use jax as backend as it is the most efficient for this task. 

The computation of the signature also profits from an optimization for GPU computation inside keras_sig. 

The KAN part implementation has been inspired from [efficient_kan](https://github.com/Blealtan/efficient-kan) and works similarly to it, thus not exactly like the [original implementation](https://github.com/KindXiaoming/pykan).

The SigKAN is a keras layers and can be used as any other keras layer, for example:

```python
import tensorflow as tf
from sigkan import SigKAN
model = Sequential([
    Input(shape=X_train.shape[1:]),
    SigKAN(100, 2, dropout = 0.), # 100 units, signature of order 2, takes an input shape (batch, sequence, features) and returns a tensor of shape (batch, sequence, 100)
    Flatten(),
    Dense(100, 'relu'),
    Dense(units=n_ahead, activation='linear')
])
```

A more complete example is provided in a notebook in the example folder.

The code is provided as is and is not specially maintained.

Please cite our work if you use this repo:

```
@article{inzirillo2024sigkan,
  title={SigKAN: Signature-Weighted Kolmogorov-Arnold Networks for Time Series},
  author={Inzirillo, Hugo and Genet, Remi},
  journal={arXiv preprint arXiv:2406.17890},
  year={2024}
}
```

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
