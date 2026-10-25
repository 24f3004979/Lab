# Preprocessing dataset

Many learning algorithms benifits from scaling, with sklearn we get a framework to build the pipeline, 
and craft the setup required fro preprocessing dataset.

### Importance of standard scaling

Implementing standard scalling with sklearn libraries for easy parameters, 
and weights management about the maping of dataset.

1. Prevents feature dominance
2. benifits gradient descent convergence
3. Helps distance based model | KNN and other models
4. Aids dimensionality reduction
5. helps in cascadial routine if the feature is already in normal distribution


### Scaling implementation with sklearn framework

Transformation of dataset could be simplified with using, transformation fit with training dataset, 
calling transformation from training maped object into testing set.

1. Make instance for required class
2. Initiate the `fit_transform` api with training dataset
3. use transform with the original transforming objeect in test pipeline

QuantileTransformer makes transformation of the data to uniform distribution without need of parameters,
might be doing insider MLE algorithms.

Taking any distribution and transforming into gausian distribution, preventing skqewnes and data simplicity.

Distribution fiting of dataset could also help into imputing values, 
with distribution the predicted values should be near perfect about original sequence in some cases ?

Normalization : A essential way to scale up each element into unit norm way, making further scaling easy
for implementig since they would retain there relative relations with original form .. ?
    Normalization is stateless, could be used into pipeline structure of the models,
    stateless -> no prior training requried for this to workout,
    normalizer.transform(X) simple usage

### Encoding Categorical Features

1. OrdinalEncoder() : used with elements with order
2. OneHotEncoder() : used for one-hot encoding features
use a categories feature with the onehot to pass out required features to pass on with the preprocessing pipeline
    
    use drop feature with binary and `n_categories` encoding
    encoder attribute : drop='if binary'
    drop='first' - geting just the n-1 encoder elements

    Infrequent categories handle
    + We can configure number of time a element should passout for encoding into a valid class,
    since one hot encoding is costly we can simply ignore the rest noise with geting rid of rest
    of non-usual elements out of the way.

3. Target Encoder | target lock zip code and regin encoding with this shot down 
    Api structure is strict here with using `fit_transform` and transform features

    use fit_transform() with this due to k-folding internal mechanism with this set, 
    using fit() function is discouraged due to its data leaking possibility and not using k-folding internal

    using into test set with endoer.transform(X_test) <-- to be used into testing pipeline
    
4. Discritization | Binning agregation feature transformation sequence
    
    using novel sequence of this transformation
    One-hot encoded discretized features can make a model more expressive, while maintaining interpretability.
    using this layered tranformation could improve the internal well

    target feature : continuous feature set for binning them | parsing through one-hot encoding feature set
    using special api with KBindsDiscretization method for the dataset to transform into rich set.
    stateless thus could be used into pipeline module stated before :)

5. Feature Binarization 
using binary filteration with the probablity layer for filtering out noise

### Imputation for missing values

+ Using imputers to fill the nan | with basic run with imputation nan columns would be auto droped
+ Need to check to keep the nan containing columns or fill with nan in better ways
+ MissingIndicator labling dataset after imputation is really good way to keep things more expressive
    Load the misssing indicator transformer for the labling for the features

    useing featureunion to group up the feature into one set | making normalization with the feature set

### Unsupervized dimentionality reduction is required for some of the features existing on core problem

### Future referencing topics

+ KernelCenterer : used for centering the phi dot product maping of kernel matrix
+ Non-Linear transformation functions : Guantile transformations and other implementations
+ B-spline based transformation with peice wise functions


### Further reading order priority 

1. [Discretization Read the docs](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization.html#sphx-glr-auto-examples-preprocessing-plot-discretization-py)
2. [Kernel Function read](https://scikit-learn.org/stable/modules/svm.html#svm-kernel)
