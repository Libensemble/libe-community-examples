
# Data-Parallel Distributed Supervised Learning with libEnsemble

`python run_libe_cnn.py -n N`

The training dataset is evenly split among the N workers. libEnsemble
starts N parallel training instances on separate, distributed
worker processes. These workers compute the gradient of the loss function for their portion
of the dataset and send their gradients to a generator process. The generator process
combines the gradients from all workers, updates the model parameters, and sends 
the updated model back to the workers.

Given the number of gradients/parameters that are being updated, `proxystore`
is used to stream the data between the processes.

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
simple example, users must only specify the number of workers `-n N` and start
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

If running multinode install `mpi4py`

### Local setup

Start a redis server instance to hold streaming data:

`redis-server`

### Multi-node

Assuming the PBS scheduler for this example.

1. Start your redis server in the background, such that it'll be 
reachable from the compute nodes:

```shell
redis-server --protected-mode no &
```

2. In `run_libe_cnn.py`, modify the `STREAMING_DATABASE_HOST` variable
to match the hostname of the login node running the redis server:

```python
STREAMING_DATABASE_HOST = "my-login-node"
```

3. Grab an interactive session on 2 nodes:

```shell
qsub -A [project] -l select=2 -l walltime=20:00 -q[queue] -I
```

4. Run the libEnsemble workflow with MPI, splitting the processes across the nodes:

```shell
mpiexec -n 4 --ppn 2 python run_libe_cnn.py
```

5. (Debugging) If the allocated GPUs aren't available, adjust the `get_device()` logic in
`gen_updates_cnn/utils.py`:

## Additional information

### Simulator

Runs model training code without optimization, while still computing
and backpropagating loss. During each training step, sends gradients across
the network to the generator, and receives updated model parameters.

### Generator

Initializes a parent model. Receives gradients from each simulator,
sums each, performs optimization to update the parent model, and sends
updated model parameters to the simulators.

### mnist directory

The CNN model training code can be run separately.

Within the `mnist` directory:

`python nn.py`.

Runs one epoch by default. See the `argparse.parser.add_argument`
calls within `main()` for additional configuration options.

`python nn_ddp.py`

Runs a similar example model, using `torch.DistributedDataParallel` for comparison
purposes.