# Preprocessing dataset

Many learning algorithms benifits from scaling, with sklearn we get a framework to build the pipeline, 
and craft the setup required fro preprocessing dataset.

## Preface with dataset

With this new era of data driven world, there are immense amount of possibilities to go with dataset.
Making intuitions and finding new things driven by question, generalizing thing for broad perspective, 
Or narrowing down for specific problem, it all just becomes exiting and fun...

### Interesting points about handling dataset
1. Explore the information with its structure
2. Try to fit statistical strcutures and framework into the dataset
3. Comming up with set of good questions about dataset the jackpot for EDA approach
4. Filter the noise and get the best clean version

### Perspective matters 

With whole feild of study centric towards making statistical structure to the dataset.
we can derive some really usefull structure and information with using our statistical tools.

- Modeling dataset through functions and transformation
- Fit certian distribution
- Extract simple frequencies of elements
- Build up abstract layers with numerical transformation to see patterns
- [ Bit advanced ] Map to different dimension to change the perspective to see dataset.



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
    stateless thus could be used into pipeline module stated before 

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


### Further reading ....

1. [Discretization Read the docs](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization.html#sphx-glr-auto-examples-preprocessing-plot-discretization-py)
2. [Kernel Function read](https://scikit-learn.org/stable/modules/svm.html#svm-kernel)


# Hands on with Sklearn implementation

Handle dataset with simple interfaces for dataset, 
+ Loaders : Loads small datasets 
+ Fetchers : fetch and load large datasets 
+ Generator : controlled syntetic dataset loading

**Data interface**
Loaders and fetchers load up bunch object a dictiionary
with data : { array of shape m,n }
with target : { array of shape n, }

Generator returns X, y as the final output

### Simple interface guide

load_*, fetch_*, make_* [loader, fetcher, generator]

iris, diabetes, digits, breast_cancer | Bundled with sklearn

make_regression, make_blobs, make_classification | data generators 
openml.org <-- public repository for loading more external datasets 
fetch_openml() : external fetch for the ml dataset

some .io api for cross tool interaction with sklearn interfaces
skimage.io -> image reference loading images into numpy arrays
scipy.io.wavfile.read reading wav file into numpy array | music processing

for managing numerical dataset, use HDF5 format -> optimized form of data handle with large numerical dataset.
    pandas does provide this format for loading this into hdf5 interfaces

### Data preprocessing with sklearn

We use sklearns tranformers for making the preprocessing pipeline, 
we use the same pipeline into training, testing and validation of the model.

With pipeline we can chain multiple transformers with uniform to make sure we apply it
to all of the three component equally.

Problems with raw dataset
1. Certain categorical data must be converted into sensible numerical representation
2. Too many features needs reduction to be usefull in a sense 
3. Missing values into the features 

Using library of transformers for data preprocessing with dataset
sklearn.preprocessing <- using the root model for the core components

sklearn.preprocessing for the preprocessing components
feature extraction : sklearn.feature_extraction
feature reduction sklearn.decomposition.pca
feature expansion : sklearn.kernel_approximation | with kernel methods of the features

### Common interface for all transformation endpoints

fit() ; learns parameters from the training dataset, stores the state of them.
transform() : using the saved feature we apply the transformation to the dataset
fit_transform() : having a upgrade with k-fold features

Feature Extraction endpoint

DictVectorizer
    converts a maping of feature name and feature values into a matrix,
    making data frame into feature matrix | how this is usefull ?
FeatureHasher : 
    high speed low memory vectorizer | using feature hashing technique 
    Need to inspect its usage and its usage guide
    used in text classification | outputs sparse matrix with scipy.sparse format

+ Feature Extraction from text and image
with both feature_extraction.image* along with other text based could have the features extraction api listing

Missing Values due to error in data capture | More things but a reality in real world
many ML algorithms dont work with NaNs, we impute the missing values to have good qaality leverage of the available dataset.

sklearn.impute
1. Simple imputer
    mean, median, most_frequent, constant | strategy='way'
    calling fit_transform() -> learning the parameter along with transformation
    we can also first build and fit all of the required elements with imputers
    add all into one pipeline

    call that pipeline with the each level of ML routiene

2. KNNImputer
    Making imputation with near by elements | Euclidien distance is used for the filling way
    parameters : n_neighbour=2, weights='uniform' <-- Adjust the weightages for computing next one :)

3. MissingIndicator  : provides the indicator for the data availability

-----

## Handling missing values

Inspect for misssing values along with there handling ways out.

### How to inspect for missing dataset

1. df.info() 
    inspect the total non-nulls 
    check for values type and convert the types -> handle errors 

2. df.isna()
    count of nan values 

3. with df[col].value_counts()
    for listing all of the possible eleemnts into the dataset 
    listing out the all possible elements into the dataset with there friq

4. df[col].unique() -> Listing all unique values from a columns
5. Finding information from other data sources into the main dataset :)

6. Drop the columns which dont add any value to the dataset at all :)

ways to drop the non-required dataset 
dropna() -> drop the row 
df.drop('column_name', in_place=True)

--- 

With simple imputer default works with column based mean filling
with KNN values the euclidian distance is used.

**OneHotEncoer**
working for encoding feature with there presense into the matrix
takes up new column filling to dictate the existance of a element in a column
we need to make sure we drop the first one to make sure we dont bloat with info, 
since one endoing could represent the existance of two :)

**LabelEncoder**
Using simle label dictionary to map out the features into thos transformation

**Ordinal Encoder**
we can list out with ordinal with orders if the feature does have such

**LabelBinaizer** | Needs clarity **

**MultiLabelBinarizer**
categorical features with values 0 and k-1, k being a number,
try this perticular transformation to facilities listing in the competition dataset

**Add a dummy feature**
Dataset agments with each value being 1,

----

## Numerical Transformation
1. Feature scaling
2. Polynomial transformation
3. Discretization

### 1.1 Feature scaling

Non-scalled features, makes iterative optimizers slow to converge, thus we
scale all of the feature into the same scale before running learning algorithms.

How to scale with sklearn api endpoints
Three scaler fcilities for numerical feature transformation

+ StandardScaler
    Transformns original to new feature, 
    scaling formula

    $$
    x^t = \frac{x - \eta }{ \sigma }
    $$
    transformer class thus need to be trained with the parameter first.


+ MaxAbsScaler 
+ MinMaxScaler scales to 0 to 1 , ranging maximum to lowest values

+ functiontransformer | construct custom transformer with this layout
    with user defined function applying the transformation
+ Polynomial transformation
    with degree value which exposes that degree of relation of the features
+ KBinsDiscretizer
    Divides continuous variable into bins
    works extreamly well with further applying ordinal or binomial encoding into this feature set.

---

### 1.2 Outlier

A data point which is significantly different from the rest of them
dealing with outliers because they are important

with df.describe() : we can fetch information about weird values if we find anything at min and max along with mean shifting
boxplots -> plt.boxplot(df['magnisium']) making visualization for the box plot
having plots outside the range with sqweezed dataset proofs the presence of outliers

Look at the distribution of the dataset, 
Important peice with ploting ploting dataset probability distribution could reveal in helping with many stuff
model that dataset with the respective probability distribution would also do wonders
for predicting rest of the missing values right ?
sns.histplot(df['magnesium'], kde=True) : ploting the smooth density function with the smooth plot

### 1.3 Methods for looking for outliers
1. difference between interquartile ranges
2. with ploting with interquartile for droping the outlier dataset

```python
data_iqr = df['col']
iqr = data_iqr.quantile(0.75) - data_iqr.quantile(0.25)
lower_bound = data_iqr.quantile(0.25) - 1.5 * iqr -- same into upper bound

data_iqr[(data above the upper limmit or bellow the lower limmit filter)][4]
```

**z-score method**:
computing how many z-score away the dataset is form the mean, 
assuming the dataset is in normal distribution
over 99 percent points are in between the 3 sigma units
With dataset following normal distribution

### 1.4 Handle outlier
caping values also helps with handling the outliers, 
replace outliers with median | robust with outliers containing dataset
transform dataset with log transform for affecting all of the dataset with skqwed ness


## Dimentionality reduction techniques

Transforming a set of feature into subset of small features,
taking down large dataset into smaller representation of features with PCA machine kernel.
+ PCA  : Linear dimentionality reduction
+ using svd to project dataset into lower dimension

with `decomposition.pca` api endpoints
with taking first k principle component,
we can alredy get the most variance of the dataset to work with
-----

## 2. Filter based feature selection

with `feature_selection` we can capture the features
Ways for feature selection

Filter based feature selection
1. VarianceThreshhold : feature must pass certain variance in the dataset
    removing features with low variance declutter the noise and make things easy

    **Univariate scoring features**
    with chisquar scoring api, F-statistics scores we select the best features out of the pool
    
    Mututalinformation : returns non-negative value | 0 -> Independent feature
    if they have high score | they have sevier dependency between both

    Chi-square scores
    compute with resppect to the label of the dataset, 
    high score feature and labels are likely correlated
    lock up with the all feature set and compare with the target to pick up the roaring features

    Most recommenede

2. SelectKbest
    use select k best with chi square -> perfect combo
    with selectkbest(chi2, k=20)
    `fit_transform(X,y)` with the target and training dataset
    20 best feature with chi-squares

    **SelectPercentile**
    With specific top 20 percentile features, with chi2, we can obtain a ratio from the dataset
    through rating of the feature scores.

3. GenericUnivariateSelect

    Craft with different ways to obtaiin the transformed featuring with this section.
    mode, param for using different modes, fpr, fdr, fwe like stuff with this ranging

Wrapper based feature relation
4. RFE
5. RFECV
6. SelectFromModel
7. SequentialFeatureSelector
8. More feature selection through kernels and trees


## 3. Wraper based feature selection 

1. Recursive feature elimination
    + Uses an estimator to recursively remove feature
    + first trains a estimator on features
    + obtains feature importance from features and removes least important feature
    + recursivly runs the elimination for obtaining desired set of features

    Use RFEC for non-knowing exact feature number which to select, 
    this runs `ref` into cross validation setup to find best feature set

    *How to use SelectFromFeature with other featuring elements like linearSVC models ?*

2. Sequential feature selection
    
    Performs feature selection with greedy approach

    + forward selection : starting from zero to optimal feature
    + backward selection : backward with all feature set

3. Applying transformation to diverse set of features

    with different data form consisting different transformers, uniting
    into single transformer unit with composing component with sklearn interface

    `sklearn.compose` with fundamental components to apply transformation to subsets and combine them.

    1. Column Transformer
        Unified interface for columns transformation

        + `column_transformation [output concatinated into one set] --> combined into single matrix`
        + usefull for applying transformation to different set of hetrogenous feature and combine them
        + combining different feature selection mechanism into a single transformation object

        Application composing with transformer sequence

        ```python 
        # Initiated sequence with the transformation
        transformer = ColumnTransformer([
            ('ageScaler', CountVectorizer(), [0]),
            ('genderEncoder', OneHotEncoding(dtype='int'), [1])
        ], 
            remainder='drop',
            verbose_feature_name_out=True
        )
        ```

    2. Transformed target regressor

        component for linear regression unit,
        making target values transform with two way transforming unit,
        could be reversed when required.
