# Sigmoid-Linear-Unit-Project
Sigmoid Linear Unit (SiLU), a variation of Gaussian Error Linear Unit (GELU), was implemented in Deep Neural Network and Convolutional Neural Network.

This project is a work on Sigmoid Linear Unit (SiLU), a variation of Gaussian Error Linear Unit (GELU), which is a high-performing neural network activation function. The GELU activation function is x(x), where (x) is the standard Gaussian cumulative distribution function for GELU. But for SiLU, (x) is the Sigmoid activation function: 1/((1+e^(-x))). SiLU is similar to a GELU in the sense that both are nonlinear and unlike a Rectified Linear Unit (ReLU), where the inputs are bordered by their sign, here the inputs are weighted by their value rather than sign.

The project was motivated from this paper: Hendrycks, Dan, and Kevin Gimpel. "Gaussian error linear units (gelus)." arXiv preprint arXiv: 1606.08415 (2016).

For DNN implementation, check: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1r1nHJ3lqaWbuynpKZDMQvNLnJkD_BVlr)

For CNN implementation, check: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EMq3fgEM714dinUbATZVWiaphBJQTOvZ)
