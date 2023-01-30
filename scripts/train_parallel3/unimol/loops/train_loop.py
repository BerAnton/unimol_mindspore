import mindspore as ms
import mindspore.ops as ops

from unimol.loss import MoleculePretrainLoss


def train_loop(model, dataset, loss_fn, optimizer):
    """This function defines training loop for model."""

    def forward(data, label):
        logits = model(data)
        loss, token_loss, coords_loss, distance_loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = ops.value_and_grad(forward, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss
    
    model.set_train(True)
    loss_history = []
    for idx, data in enumerate(dataset.create_tuple_iterator()):
        sample, target = data[:4], data[4:]
        loss = train_step(sample, target)
        loss_history.append(loss)
    mean_epoch_loss = sum(loss_history) / len(loss_history)
    return mean_epoch_loss
    