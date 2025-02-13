
# Data-Parallel Distributed Supervised Learning with libEnsemble

`python run_libe_cnn.py -n N`

The training dataset is evenly split among the N workers. libEnsemble
starts N parallel training instances on separate, distributed
worker processes. These workers compute the gradient of the loss function for their portion
of the dataset and send their gradients to a generator process. The generator process
combines the gradients from all workers, updates the model parameters, and sends 
the updated model back to the workers.

Given the number of gradients/parameters that are being updated, `proxystore`
is used.

## Justification

If the local dataset for training a single model must remain small,
then time-to-generalization during training may be high as batches must be loaded
and unloaded. Data-parallel or cross-node techniques in many AI libraries are difficult
to run on multi-node systems, or assume user familiarity with parallel programming.

libEnsemble allows easy parallelization across multiple nodes and HPC
systems. "Ensembles" of experiments with libEnsemble take no configuration
to run on separate nodes and intercommunicate. 

In this example, one model is optimized in parallel by a parent "generator"
model based on the summed gradients from the N "simulator" models. For this
simpler example, users must only specify the number of workers `-n N` and start
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
the network to the generator running a training loop. Parameters from the generator
are updated inside each training loop.

## Generator

Initializes a parent model. Receives gradients from each simulator,
sums each, performs optimization to update the model, and sends
updated model parameters to the simulators.

## mnist directory

The CNN model training code can be run separately.

Within the `mnist` directory:

`python nn.py`.

Runs one epoch by default. See the `argparse.parser.add_argument`
calls within `main()` for additional configuration options.
