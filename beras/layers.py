import numpy as np

from typing import Literal
from beras.core import Diffable, Variable, Tensor

DENSE_INITIALIZERS = Literal["zero", "normal", "xavier", "kaiming", "xavier uniform", "kaiming uniform"]

class Dense(Diffable):

    def __init__(self, input_size, output_size, initializer: DENSE_INITIALIZERS = "normal"):
        self.w, self.b = self._initialize_weight(initializer, input_size, output_size)

    @property
    def weights(self) -> list[Tensor]:
        return self.w, self.b

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for a dense layer! Refer to lecture slides for how this is computed.
        """   
        # x  * w + b
        # (num_samples, input_size) * (input_size, output_size) + (1, output_size)
        p1 = x @ self.w
        out = np.apply_along_axis(lambda sample: sample + self.b, 1,p1)
        return out

    def get_input_gradients(self) -> list[Tensor]:
        input_grad = self.w # Gradient wrt inputs
        return [input_grad] # return as list

    def get_weight_gradients(self) -> list[Tensor]:
        grad_w  = np.expand_dims(self.inputs[0], axis=2)
        grad_b = np.ones_like(self.b)
        return [grad_w, grad_b]
    
    @staticmethod
    def _initialize_weight(initializer, input_size, output_size) -> tuple[Variable, Variable]:
        """
        Initializes the values of the weights and biases. The bias weights should always start at zero.
        However, the weights should follow the given distribution defined by the initializer parameter
        (zero, normal, xavier, or kaiming). You can do this with an if statement
        cycling through each option!

        Details on each weight initialization option:
            - Zero: Weights and biases contain only 0's. Generally a bad idea since the gradient update
            will be the same for each weight so all weights will have the same values.
            - Normal: Weights are initialized according to a normal distribution.
            - Xavier: Goal is to initialize the weights so that the variance of the activations are the
            same across every layer. This helps to prevent exploding or vanishing gradients. Typically
            works better for layers with tanh or sigmoid activation.
            - Kaiming: Similar purpose as Xavier initialization. Typically works better for layers
            with ReLU activation.
        """

        initializer = initializer.lower()
        assert initializer in (
            "zero",
            "normal",
            "xavier",
            "kaiming",
        ), f"Unknown dense weight initialization strategy '{initializer}' requested"

        #### start code ####

        bias = Variable(np.zeros((output_size,)))  # The bias weights should always start at zero.

        if initializer == "zero":
            weights = Variable(np.zeros((input_size, output_size)))

        elif initializer == "normal":
            weights = Variable(np.random.normal(0, 1, (input_size, output_size)))  # N(0,1)

        elif initializer == "xavier":
            # From tf.keras.initializers.GlorotNormal description:
            # Draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in + fan_out)) 
            # where fan_in is the number of input units in the weight tensor 
            # fan_out is the number of output units in the weight tensor.
            stddev = np.sqrt(2 / (input_size + output_size))
            weights = Variable(np.random.normal(0, stddev, (input_size, output_size)))


        elif initializer == "kaiming":
            stddev = np.sqrt(2 / input_size)
            weights = Variable(np.random.normal(0, stddev, (input_size, output_size)))  

        return weights, bias
