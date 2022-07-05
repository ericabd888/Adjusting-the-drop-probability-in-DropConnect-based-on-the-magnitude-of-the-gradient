# Training a neural network by adjusting the drop probability in DropConnect based on the magnitude of the gradient
## Abstract
In deep learning, dropout is a regularization technique which is often used to solve overfitting, by randomly discarding neurons with a fixed probability during training step, so that each neuron does not depend too much on other neurons from each other, thereby improving the generalization ability of the model.

DropConnect applies dropout to the weights and biases which is connecting these neurons, by randomly discarding these weights and biases with a fixed probability during training, rather than on neurons.

This paper proposes a new model Gradient DropConnect, which is an improvement of dropConnect, the model will use the gradient of each weight and bias to determine each probability that weights and biases are discarded. Through experiments, we found that by increasing the the discarding probability with a smaller gradient of weight and bias can effectively solve the problem of overfitting.
