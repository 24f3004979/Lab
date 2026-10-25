# Data Loading with sklearn

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
    x^! = \frac{x - \mue}{\sigm}
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

## Outlier

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

### Methods for looking for outliers
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

### Handle outlier
caping values also helps with handling the outliers, 
replace outliers with median | robust with outliers containing dataset
transform dataset with log transform for affecting all of the dataset with skqwed ness


-----

## Filter based feature selection

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
