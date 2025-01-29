import numpy as np
from libensemble.message_numbers import PERSIS_STOP, STOP_TAG, EVAL_SIM_TAG, WORKER_DONE
from libensemble.specs import output_data, input_fields, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport as ToGenerator


from mnist.nn import main as run_cnn


def train_model(self, args, device, train_loader, optimizer, epoch):
    self.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = self(data)
        loss = F.nll_loss(output, target)
        self.total_train_loss += loss
        loss.backward()
        # UPDATE GRADIENTS FROM GEN HERE
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def _run_cnn_send(generator, sim_specs, summed_gradients=None):

    Output = np.zeros(1, dtype=sim_specs["out"])

    grads = run_cnn(summed_gradients)
    Output["local_gradients"] = [i.cpu().detach().numpy() for i in grads]
    generator.send(Output)


@input_fields(["summed_gradients"])
@persistent_input_fields(["summed_gradients"])
@output_data(
    [("local_gradients", object, (8,))]
)
def mnist_training_sim(H, _, sim_specs, info):

    generator = ToGenerator(info, EVAL_SIM_TAG)

    _run_cnn_send(generator, sim_specs, None)

    while True:
        tag, Work, calc_in = generator.recv()
        if tag in [PERSIS_STOP, STOP_TAG]:
            break

        _run_cnn_send(generator, sim_specs, calc_in["summed_gradients"])

    return None, {}, 0