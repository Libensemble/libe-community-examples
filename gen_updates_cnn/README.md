
# Distributed CNN Optimization with libEnsemble

`python run_libe_cnn.py -n N`

Starts N parallel CNN training instances on separate, distributed
worker processes. The workers send their parameters to a manager process,
where optimization is performed on the combined data. Updated model
weights are sent back to the workers.

## Setup

`pip install -e .` in this directory.

If using pixi, `pixi shell`.

## Simulator

Runs `minst/nn.py`'s model training code (`main()`) and sends
the training loss, model parameters, and last layer's gradient to
the generator. Subsequent runs reinitialize the model based on
new weights from the generator.

## Generator

Receives training loss, model parameters, and gradient from each worker,
sums each, performs optimization, and sends updated model weights back
to each worker.

## mnist directory

The CNN model training code can be run separately.

Within the `mnist` directory:

`python nn.py`.

Runs one epoch by default. See the `argparse.parser.add_argument`
calls within `main()` for additional configuration options.