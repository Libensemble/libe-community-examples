
# Distributed CNN Optimization with libEnsemble

`python run_libe_cnn.py -n N`

Starts N parallel CNN training instances on separate, distributed
worker processes. The workers send their gradients during training
to a manager process, where the combined gradients are optimized
into updated model weights. These are sent back to the workers.

The dataset is evenly split among the N workers.

Gradients are streamed across the network, including across nodes.

## Justification

If the local dataset for training a single model must remain small,
then time-to-generalization during training may be high as batches must be loaded
and unloaded. Data-parallel or cross-node techniques in many AI libraries are difficult
to run on multi-node systems, or assume user familiarity with parallel programming.

libEnsemble allows easy parallelization across multiple nodes and HPC
systems. "Ensembles" of experiments with libEnsemble take no configuration
to run on separate nodes and intercommunicate. 

In this example, N models are optimized in parallel by a parent "generator"
model based on the summed gradients from the N "simulator" models. For this
simpler example users must only specify the number of workers `-n N` and start
a background redis server for data-streaming.

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

### Local setup

Start a redis server instance to hold streaming data:

`redis-server`

### Multi-node

TODO

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