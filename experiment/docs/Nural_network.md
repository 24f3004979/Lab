# Ideas with deep learning

A new domain of prediction which could learn patterns, out of dataset and 
return a really good maping function from given set of input to output.

We train parameters which best defines a function to map the two domains.
Inspired by biological nural network, deep learning consists of layers as the name.

With each nurons firing its computation and connected to other, we see ANN as acyclic graph of functions, 
linked with weights and biases, which are the parameters model learns to tweak its projecting outputset.

## Components
1. Input layer

Interface where data vectors enters into the subsequent branches,
with interaction with weights [ random at initial stages - gets trained through learning algorithm,
to make sure they provide certain weightages to certain feature for further layers]

2. Hidden layer

Generic functions initiated with their parameters, which gets finetuned to work best for its inputsets,
getin filtered through the learning algorithm along the training way.

3. Activation functions

A non-linear function specifically desinged for fine tuning the internal data of the model.
Does the affine transformation helping the internal function a appropriate dataset for their functions.
they work to transform the dataset to make easy for further follow up nurons to work well.

4. Output Layer

Final interception of the nurons, and final transformation to the output space, and desired output range targeting.
it aligns the input from previous hidden layer to align with the problem.

