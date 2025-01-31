
# Distributed CNN Optimization with libEnsemble

`python run_libe_cnn.py -n N`

Starts N parallel CNN training instances on separate, distributed
worker processes. The workers send their gradients to a manager process,
where optimization is performed on the combined data. Updated model
weights are sent back to the workers.

## Setup

### Dependencies

If using pixi, `pixi shell`.

Otherwise, install:

```
torch = ">=2.5.1, <3"
torchvision = ">=0.20.1, <0.21"
proxystore = ">=0.8.0, <0.9"
redis = ">=5.2.1, <6"
```

Then start a redis server instance to hold streaming data:

`redis-server`

## Simulator

Runs model training code without optimization. Sends
the last layer's gradient to the generator. Subsequent
runs reinitialize the model using optimized parameters
from the generator.

## Generator

Initializes a parent model. Receives gradients from each simulator,
sums each, performs optimization using parent model, and sends
updated model parameters to the simulators.

## mnist directory

The CNN model training code can be run separately.

Within the `mnist` directory:

`python nn.py`.

Runs one epoch by default. See the `argparse.parser.add_argument`
calls within `main()` for additional configuration options.