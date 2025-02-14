import numpy as np

from .core import Diffable,Tensor

class Activation(Diffable):
    @property
    def weights(self): return []

    def get_weight_gradients(self): return []


################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Activation):

    ## TODO: Implement for default intermediate activation.

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def forward(self, x) -> Tensor:
        """Leaky ReLu forward propagation!"""
        self.inputs = x
        return np.maximum(self.alpha * x, x)

    def get_input_gradients(self) -> list[Tensor]:
        """
        Leaky ReLu backpropagation!
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        grad = np.ones_like(self.inputs)
        grad[self.inputs < 0] = self.alpha
        return [grad]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J

class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self):
        super().__init__(alpha=0)


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Activation):
    
    ## TODO: Implement for default output activation to bind output to 0-1
    
    def forward(self, x) -> Tensor:
        self.inputs = x
        self.outputs = Tensor(1 / (1 + np.exp(-x)))
        return self.outputs

    def get_input_gradients(self) -> list[Tensor]:
        """
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        grad = self.outputs * (1 - self.outputs)
        return [Tensor(grad)]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J


class Softmax(Activation):
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    ## TODO [2470]: Implement for default output activation to bind output to 0-1

    def forward(self, x):
        """Softmax forward propagation!"""
        ## Not stable version
        ## exps = np.exp(inputs)
        ## outs = exps / np.sum(exps, axis=-1, keepdims=True)

        ## HINT: Use stable softmax, which subtracts maximum from
        ## all entries to prevent overflow/underflow issues
        shift_x = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(shift_x)
        self.outputs = Tensor(exps / np.sum(exps, axis=-1, keepdims=True))
        return self.outputs

    def get_input_gradients(self):
        """Softmax input gradients!"""
        x, y = self.inputs + self.outputs
        bn, n = x.shape
        grad = np.zeros(shape=(bn, n, n), dtype=x.dtype)
        
        # TODO: Implement softmax gradient
        x, y = self.inputs[0], self.outputs
        bn, n = x.shape
        grad = np.zeros((bn, n, n), dtype=x.dtype)
        for b in range(bn):
            diag = np.diag(y[b])
            outer = np.outer(y[b], y[b])
            grad[b] = diag - outer
        return [grad]
