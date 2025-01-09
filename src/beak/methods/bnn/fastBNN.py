from typing import List, Tuple, Sequence, Dict, Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tqdm import tqdm
from beak.utilities.transform import __transform_minmax


def set_global_seed(seed: int) -> None:
    """
    Set the global random seed for reproducibility.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_global_seed(42)


def initializer(shape: Tuple[int, int]) -> tf.Tensor:
    """
    Xavier initialization for weights.
    """
    return tf.random.truncated_normal(
        shape,
        mean=0.0,
        stddev=np.sqrt(2 / sum(shape))
    )


def create_layer_params(d_in: int, d_out: int) -> Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable]:
    """
    Create layer parameters for weights and biases with initialization.
    """
    w_loc = tf.Variable(initializer((d_in, d_out)), name="w_loc")
    w_std = tf.Variable(initializer((d_in, d_out)) - tf.constant(6.0), name="w_std")
    b_loc = tf.Variable(initializer((1, d_out)), name="b_loc")
    b_std = tf.Variable(initializer((1, d_out)) - tf.constant(6.0), name="b_std")

    return w_loc, w_std, b_loc, b_std


def dense_layer_forward(
    x: tf.Tensor,
    params: Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable],
    trainable: bool = True
) -> tf.Tensor:
    """
    Forward pass through a dense layer with optional perturbations.
    """
    w_loc, w_std, b_loc, b_std = params
    w_sigma, b_sigma = tf.nn.softplus(w_std), tf.nn.softplus(b_std)

    if trainable is True:
        s = tfp.random.rademacher(tf.shape(x))
        w_r = tfp.random.rademacher([tf.shape(x)[0], tf.shape(w_loc)[1]])
        b_r = tfp.random.rademacher([tf.shape(x)[0], tf.shape(b_loc)[1]])

        w_eps = tf.random.normal(tf.shape(w_loc))
        b_eps = tf.random.normal(tf.shape(b_loc))

        w_samples = w_sigma * w_eps
        b_samples = b_sigma * b_eps

        w_perturb = w_r * tf.matmul(x * s, w_samples)
        w_outputs = tf.matmul(x, w_loc) + w_perturb
        b_outputs = b_loc + b_r * b_samples

        return w_outputs + b_outputs
    else:
        return tf.matmul(x, w_loc) + b_loc


def kl_divergence(parameter: Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable]) -> tf.Tensor:
    """
    Calculate KL divergence for a single layer.
    """
    w_loc, w_std, b_loc, b_std = parameter
    weight = tfd.Normal(w_loc, tf.nn.softplus(w_std))
    bias = tfd.Normal(b_loc, tf.nn.softplus(b_std))
    prior = tfd.Normal(0.0, 1.0)
    return tf.reduce_sum(tfd.kl_divergence(weight, prior)) + tf.reduce_sum(tfd.kl_divergence(bias, prior))


def build_network(
    input_dims: List[int],
    output_dims: List[int]
) -> List[Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable]]:
    """
    Build a network by creating parameters for each layer.
    """
    return [create_layer_params(d_in, d_out) for d_in, d_out in zip(input_dims, output_dims)]


def forward_network(
    x: tf.Tensor,
    layers: List[Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable]],
    activations: List[str],
    trainable: bool = True
) -> tf.Tensor:
    """Forward pass through a sequence of layers with activation functions."""
    for params, activation in zip(layers, activations):
        x = dense_layer_forward(x, params, trainable)
        x = apply_activation(x, activation)
    return x


def apply_activation(x: tf.Tensor, activation: str) -> tf.Tensor:
    """
    Apply specified activation function.
    """
    return tf.keras.activations.get(activation)(x)


def calculate_total_kl(layers: List[Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable]]) -> tf.Tensor:
    """
    Compute the total KL divergence loss for all layers.
    """
    return tf.reduce_sum([kl_divergence(layer) for layer in layers])


def build_bayesian_network(
    input_shape: int,
    core_units: List[int],
    head_units: List[int],
) -> Tuple[List, List, List]:
    """
    Build a Bayesian network consisting of one core network and two heads (loc and std).
    """
    head_units = head_units + [1]

    core_unit = build_network([input_shape] + core_units[:-1], core_units)
    loc_head = build_network([core_units[-1]] + head_units[:-1], head_units)
    std_head = build_network([core_units[-1]] + head_units[:-1], head_units)
    return core_unit, loc_head, std_head


def forward_layers(
    x: tf.Tensor,
    layers: List[Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable]],
    activations: List[str],
    trainable: bool = True
) -> tf.Tensor:
    """
    Forward pass through a list of layers with corresponding activations.
    """
    for params, activation in zip(layers, activations):
        x = dense_layer_forward(x, params, trainable)
        x = apply_activation(x, activation)
    return x


def bayesian_network_forward(
    x: tf.Tensor,
    core_unit: List,
    loc_head: List,
    std_head: List,
    trainable: bool = True
) -> tf.Tensor:
    """
    Forward pass through the Bayesian network (core + loc and std heads).
    """
    activation = "relu"

    # Core layers
    core_activations = [activation] * len(core_unit)
    core_output = forward_layers(x, core_unit, core_activations, trainable)

    # Apply activation to all but the last layer
    loc_activations = [activation] * (len(loc_head) - 1) + [None]
    std_activations = [activation] * (len(std_head) - 1) + [None]

    # Forward pass through loc and std heads
    loc_preds = forward_layers(core_output, loc_head, loc_activations, trainable)
    std_preds = forward_layers(core_output, std_head, std_activations, trainable)

    return tf.concat([loc_preds, tf.nn.softplus(std_preds)], axis=1)


def negative_log_likelihood(y_true: tf.Tensor, y_pred: tf.Tensor):
    """
    Calculate negative log likelighood.
    """
    loc = y_pred[:, 0]
    scale = y_pred[:, 1]
    distribution = tfd.Normal(loc=loc, scale=scale)
    return -tf.reduce_mean(distribution.log_prob(y_true[:, 0]))




def fit_model(
    data_train,
    input_shape: int,
    core_units: List[int],
    head_units: List[int],
    lr: float,
    epochs: int,
    N: int,
) -> Tuple[Tuple[List, List, List], np.ndarray]:
    """
    Train the Bayesian density network and return the trained model parameters and metrics.
    """
    core_layers, loc_head, std_head = build_bayesian_network(
        input_shape,
        core_units,
        head_units,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def train_step(x_data: tf.Tensor, y_data: tf.Tensor, N: tf.Tensor) -> tf.Tensor:
        """
        Perform a single training step (forward pass + loss calculation + optimization).
        """
        with tf.GradientTape() as tape:
            y_pred = bayesian_network_forward(x_data, core_layers, loc_head, std_head, trainable=True)
            kl_loss = calculate_total_kl(core_layers + loc_head + std_head)
            elbo_loss = (kl_loss / N) + negative_log_likelihood(y_data, y_pred)

        variables = [v for layer in core_layers + loc_head + std_head for v in layer]
        gradients = tape.gradient(elbo_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return elbo_loss

    elbo = np.zeros(epochs)
    N = tf.constant(N, dtype=tf.float32)

    for epoch in tqdm(range(epochs), desc="Training", unit="Epoch"):
        for x_data, y_data in data_train:
            elbo[epoch] = train_step(x_data, y_data, N).numpy()

    return (core_layers, loc_head, std_head), elbo


def predict_model(
    model: Tuple[List, List, List],
    data: np.ndarray,
    target_shape: Optional[Tuple[int, ...]] = None,
    clip_pred: bool = True,
    norm_sd: bool = True,
) -> List:
    """
    Generate predictions from the trained Bayesian density network.
    """
    core_layers, loc_head, std_head = model
    x_data = tf.convert_to_tensor(data, dtype=tf.float32)

    y_pred = bayesian_network_forward(x_data, core_layers, loc_head, std_head, trainable=False)
    prediction_mean = y_pred[:, 0].numpy()
    prediction_sd = y_pred[:, 1].numpy()

    if target_shape is not None:
        prediction_mean = prediction_mean.reshape(target_shape)
        prediction_sd = prediction_sd.reshape(target_shape)

    return [
        np.clip(prediction_mean, 0, 1) if clip_pred is True else prediction_mean,
        __transform_minmax(prediction_sd) if norm_sd is True else prediction_sd
    ]
