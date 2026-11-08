# Doubts index 💀

1. When we explain the architechture in terms of those crosses and numbers what do each of them specify each ?

ANN : A bunch of singleton processors working in sync for solving one problem for which we can't code out algorithmic approach, with abundance of dataset we can direct multiple perceptron units to allign to solve our problem could be paired with other ways too for training more complicated and sophesticated network.


With treditional linear model being able to solve for simple regression task, we learnt kernel trick could extend the horizon more, and it works good for generalizing into the train set but fails to optimize for global spatiality in test set, since genral kernels are made for local smoothness not to inherit special feature out of dataset.

We can either craft out specific kernels taking so much time to craft into and train the linear model into that, or we can choose from set of kernel function which in our case deep learning model can fine tune on its own over its training phase.

With simplest to start with, we set up the ground with Linear models, we utilise composition with multiple functions working in paralell for completing a task, with multilayer perceptron we can define differnt layers along with special layering called non-linear activation function which takes the linear function to next level to learn real world non-linearity from the dataset,

we have more paradigm like CNN solving specific nail point of problem with deep learning.

Overview of simplest model flow end to end.
feed forward neural network is bunch of processors combined together where we initiate weights randomly and model trains to learn about the pattern overtime, 

for example : Feed forward neural network
1. Input layer
    Defined with dimension of data we would consider and input batch which we are passing through,
    it could be made to wrk in batches with big dataset or take all at once
    `input_size x input_neuron` 
2. Hidden layers
    Add with caution to not overfit based on choice and architechture which we choose, 
    along with complexity of the model. 
    I perticularly dont understand how much hidden layer to add how to define there shapes flow ?
    `input_neuron_number x h1_size` these would be chianed weights which are combined with bias term which are the linear models whse output are transfered to the next layer after activation function passout.

3. Output layer : it takes the final hidden layer output for the final nuron map

With these processing we complete one cycle of feed forward neural network, after which we compute the loss and with that loss model computes something called back prop which is tracing back errors to weight for adjusting it, dont know its math perticulary but its differentiation chained back to each weights for there adjustment for performing gradient descent like feature for soo many differnet function, torch does it automatically we are blessed.

so thats how the thing is in my brain fog.

---

2. What is the role of dense layer ?

    A fully connected layers where each nuron is connected to rest of them, used for aggregating features after a running layer, we combine features output of all of the previous for last one.


3. How to calculate total trainable parameters in neural network

    Use simple rules :)

4. How can whole deep learning framework could be contained with simple statements ?
5. Back propagation needs real attention to fully understand the neural network

## Experimental listing
+ Make a simple neural network implementation with hand written without using torch with just numpy.
+ Inspect a image model with its internal filter layer learning status and its learning pattern


## Reading references
+ [MLP](https://arxiv.org/pdf/2203.14679)
+ [DL](https://arxiv.org/pdf/1404.7828) [red]
+ [Back propagation](https://arxiv.org/pdf/2301.09977)
+ [CNN](https://arxiv.org/pdf/1511.08458)
+ [Image Net](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
+ [GoogleNet](https://arxiv.org/pdf/1409.4842)
