
# Distributed CNN Optimization with libEnsemble

`python run_libe_cnn.py -n N`

Starts N parallel CNN training instances on separate, distributed
worker processes. The workers send their parameters to a manager process,
where optimization is performed on the combined data. Updated model
weights are sent back to the workers.

## Setup

If using pixi, `pixi shell`.

Otherwise, install:

```
torch = ">=2.5.1, <3"
torchvision = ">=0.20.1, <0.21"
```

## Simulator

Runs `minst/nn.py`'s model training code. Sends
the training loss, model parameters, and last layer's gradient to
the generator. Subsequent runs reinitialize the model using
optimized weights from the generator.

## Generator

Receives training loss, model parameters, and gradient from each worker,
sums each, performs optimization, and sends optimized model parameters back
to each worker.

## mnist directory

The CNN model training code can be run separately.

Within the `mnist` directory:

`python nn.py`.

Runs one epoch by default. See the `argparse.parser.add_argument`
calls within `main()` for additional configuration options.