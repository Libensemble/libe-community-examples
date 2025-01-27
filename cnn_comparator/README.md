Both of these work in parallel on Perlmutter and uses GPUs on one node, but does not work across nodes.

pytorch_cnn_NCCL_parallel.py: Parallelize with torch.distributed.all_reduce
pytorch_cnn_NCCL_with_DDP.py: Parallelize with torch.nn.parallel.DistributedDataParallel

**pytorch_cnn_NCCL_parallel.py**
libEnsemble would send to generator to do the all reduce where it does
dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

**pytorch_cnn_NCCL_with_DDP.py**
Implicit parallelism.

