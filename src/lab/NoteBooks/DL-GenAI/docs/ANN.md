# Artificial Nural Network and deep learning

Fundamental questions
1. What really those nurons means at all ?
2. How does perceptrons fit into this picture
3. What other then the cool picture i can derive and experiment with ?
4. How does layers are connected ?
	In which basis does we increase the layers number,
	How we would inspect the nuron setup ?

5. Where does the architechture things comes into play with nurons ?
6. What really the connection means in this sense ?

Overview approach with the nuron structure
ANN is a smart function which can connect abstract numbers to some other numbers of interest
Maping from input layer to the output through functions abstractions throughout the journey.

## Computations inside the neural network

**step1. Linear part**
sum of all number + bias = z
z : linear combination of input with bias

z passed through linear activation function g(z) -> a [final output number]

Fundamental Building block
Linear Sum ==> Non-Linear activation function [ core activation function which learns weights ]

connection logic : with giving the next functions input the output of current function we chain multiple functions together
with nurons modeled with weights and bias along the way.

+ Need to read docs with nurons and their architechture
+ Model the learnt thing into paper with all layes and working mechanism with a modeled problem to solve with
+ try to derive things for Linear regression task

---

Nural network working mechanism

1. input vectors
2. weights labeled for each connection [ numbers matrix ]
3. each node consist of a bias number with it 
4. compute the matrix multiplication of weights with all input elements
5. z = weights X x_input + bias
6. compute the activation function output with the z as an input to the function

Thats the final output which we get through one nural network working end to end :)
How ? 
+ Does the weights are transfering learning ?
+ what does the activation functions are really doing although rather then having more overhead for calculations ?

Stacking Linear layer is useless, 
All linear fuunctions stack up to become one linear function
* without non-linear activation function a deep nural network colapses under one single model irrespective of
using soo many layers

Glimpse for the layering
We are layering the input spectrum to further numering system :)
L1 = (W_1 X x_input) + bias_1 
L2 = W_2 X (g(L1)) + bias_2

+ What do we gain with non-linear functions used into the network ?
with non-linear we can learn complex stuff ? but how what this really emply at ground ?

## Most used nural networks
Sigmoidal nural networks
- Nice function for binary classification tasks
- saturation points makes thing simple for all ranges

Problems
Again i am writing without any mathematical context related to deep nural network
- Vanishing gradients - stops learning for deep nural networks

ReLu activation function
1. computationaly efficient | simple threshhold calculation 
2. No vanishing gradients | for x > 0 constant gradients -> allows for deep propagation of weights
3. Sparcity with zero output

Dying Network problem
with removing negetive outputs, if we have any where negetive then it wont recover
into starting point we can see if it goes to negetive all nurals outputs zero :)
- No zero centered - outputs are always positive - problem for nural network
+ i am geting no mathematical sense with nural networks :)

Dying ReLU fix
+ with leaky ReLu
when z is small then zero outputs a alpha X z which is small 0.001 X z value 
helping for further propagation of weigths

GELU
Gaussian Error Linear unit
probabilistic alternative to relu

Rules for hidden and activation functions
1. Use RELU | Dont use sigmoid or tanh due to vanishing gradient problems
2. based on taks use for output layer, Binary [ sigmoidal ] , Multiclass softmax
