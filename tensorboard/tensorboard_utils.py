
def weight_histograms_conv2d(writer, step, weights, name):
    weights_shape = weights.shape
    num_kernels = weights_shape[0]
    for k in range(num_kernels):
        weights = weights[k]
        tag = f"weights/layer_{name}/kernel_{k}"
        writer.add_histogram(tag, weights.flatten(), global_step=step, bins='tensorflow')
        tag = f"gradients/layer_{name}/kernel_{k}"
        writer.add_histogram(tag, weights.grad.detach().flatten(), global_step=step, bins='tensorflow')


# rozgranicz na rozne komórki
def weight_histograms_rnn(writer, step, layer, name):
    weight_hh_l0 = layer.weight_hh_l0
    tag = f"weights/rnn/hidden_layer_{name}"
    writer.add_histogram(tag, weight_hh_l0.flatten(), global_step=step, bins='tensorflow')
    tag = f"gradients/rnn/hidden_layer_{name}"
    writer.add_histogram(tag, weight_hh_l0.grad.detach().flatten(), global_step=step, bins='tensorflow')

    weight_ih_l0 = layer.weight_ih_l0
    tag = f"weights/rnn/input_layer_{name}"
    writer.add_histogram(tag, weight_ih_l0.flatten(), global_step=step, bins='tensorflow')
    tag = f"gradients/rnn/input_layer_{name}"
    writer.add_histogram(tag, weight_ih_l0.grad.detach().flatten(), global_step=step, bins='tensorflow')


def weight_histograms_linear(writer, step, weights, name):
    tag = f"weights/layer_{name}"
    writer.add_histogram(tag, weights.flatten(), global_step=step, bins='tensorflow')
    tag = f"gradients/layer_{name}"
    writer.add_histogram(tag, weights.grad.detach().flatten(), global_step=step, bins='tensorflow')


def get_val_data(dataset, attr: str, external_ids = None):
    if external_ids:
        data = dataset.dataset.__getattribute__(attr)[dataset.ids][external_ids]
    else:
        data = dataset.dataset.__getattribute__(attr)[dataset.ids]
    return data.unsqueeze(1) if data.dim() == 3 else data