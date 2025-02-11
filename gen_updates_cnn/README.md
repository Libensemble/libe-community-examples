
# Distributed CNN Optimization with libEnsemble

`python run_libe_cnn.py -n N`

Starts N parallel CNN training instances on separate, distributed
worker processes. The workers send their gradients during training
to a manager process, where the combined gradients are optimized
into updated model weights. These are sent back to the workers.

The dataset is evenly split among the N workers.

The gradients are streamed across the network

## Justification

Optimization-in-the-training-loop is a common paradigm for training AI models.
However, if the local dataset for a model must remain small due to memory constraints,
then time-to-generalization for the model may be high as batches of data are read into
and out of memory for training. Furthermore, data-parallel techniques included in many
common AI python libraries have difficulties running on/across HPC systems.

libEnsemble is developed for easy parallelization on and across multiple nodes and HPC
systems. Worker processes take no additional configuration to run on separate nodes
and communicate with a head node.

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

Runs model training code without optimization. Sends gradients across
network to generator inside training loop. Parameters from generator
are updated inside each training loop.

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