# Deep Feedforward Network

Learning patterns with the help of nurons alike mathematical models,which computes in layers,
feeds information to the forward layers to learn patterns from the dataset.

Composed result of many functions working together, with acyclic graph based flow.
target to best approximate the underlying pattern $ f^* $ function.

Nural network, is acyclic graph of functions which works in layers, to train parameters
for best approximating the underlying pattern in given dataset.FeedForward neural network
works with forwarding information flow with intermediate computations,
when computations are re-fed to the network its **recurrent nural network**.

## 1.1 Emergence of idea of nural networks

We can set ground with understanding scope of linear models, they work really well
with linear space and linear hyperplanes, with non-linear dataset we need to fix them, 
take them to the same structure to be able to use the linear functions.

With $\phi(x)$ being the mapping(rbf) to map features to infinite dimensional space for extracting the linear relations, such that we can apply the linear function.

**How to get the mapping $\phi$**
1. Using generic mapping functions

    + We can generalize well with training set
    + Testing set cant be easily generalized
    + focused on local smoothness of function
    + Non-encoding of enough prior information
2. Manual Engineer the mapping function

    + Needs domain specific skills
    + Required dedicated long term time
    + deligative task to perform

3. **Lock targets to learn**

    Being highly generic and specific with learning the parameter, <br>
    with training we get the specific maping functions out of all generic functions options.


Example : Learning **XOR** function,
with simple XOR maping we use non-linear function to map the input set into specific space for linear model to work with,
to make proper maping of XOR subset of information.

using non-linear function to describe features, With most nural networks does with, 
affine transformation through learned parameters,
with fixed non-linear activation functions called as activation function.

with the need of non-linear activation function in between we transform the feature space into a space where linear model could map out to the target well then before
We use non-linear functions because they could help to capture non-linear functions, help to map out to relevent spaces for linear function to workwith, 

Using multiple linear function in layers simply means using one due to mathematical property of collapsing into one linear function equivalency.

## 1.2 Gradient-Based Learning

## More reading
+ [Gradient Based learning | 172 PGN ](https://www.deeplearningbook.org/contents/mlp.html)