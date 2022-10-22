# MLP from Scratch
### I made this project in order to get a better understanding in Deep Learning. 
<br> 

Note: I am not really into C++, so the codes might be seen inefficient to some C++ experts


## About

- The model is coded in C++
- No external library is used during the proccess
- As task I choosed "**XOR Gate Learning**", which is one of the basis task for deep learning models 
- The activation function is the "**Sigmoid**" function for all layers
- The model contains:
  - 1 input layer with 2 neurons
  - 1 hidden layer with 4 neurons
  - 1 output layer with 1 neuron
- Weights are initiliazed respect to "**Normal Distribution**"
<br>

## Results

- Sample number: **80** (4*20)
- Training duration: **6.07392 secs** 
- Epoch number: **5000**
- Learning rate: **0.007**
- Loss: **0.000108905**
- Accuracy: **%97.674**
- Loss / Accuracy Graph: <br> <br>
![This is an image](https://github.com/IsmailKonak/MLP-From-Scratch-CPP/blob/main/XOR_cpp_loss_accuracy.png)

<br>
<br>

## Reference

I have benefited greatly from the following resources in this process:
- [Medium](https://medium.com/@tiago.tmleite/neural-networks-multilayer-perceptron-and-the-backpropagation-algorithm-a5cd5b904fde) - Neural Networks, Multilayer Perceptron and the Backpropagation Algorithm by _Tiago M. Leite_

- [Youtube](https://www.youtube.com/watch?v=tIeHLnjs5U8) - Backpropagation calculus | Chapter 4, Deep learning

- [Youtube](https://www.youtube.com/watch?v=sDv4f4s2SB8) - Gradient Descent, Step-by-Step

- [Youtube](https://www.youtube.com/watch?v=IHZwWFHWa-w) - Gradient descent, how neural networks learn | Chapter 2, Deep learning
- Book - Deep Learning, **_Ian Goodfellow_** and **_Yoshua Bengio_** and **_Aaron Courville_**
