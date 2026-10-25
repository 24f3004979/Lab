## Brief note about scikit learn library

This library is lined up with machine learning framework,

API lined up for each core component of machine learning
1. Training data and preprocessing 
2. Model building and optimization procedure
3. model selection and evalutation
4. Model inspection

### API design principle basics

With simple consistent interfaces and great resilient work in the hood

+ All api share same consistent interface
+ Inspection learnable parameters are well exposed for inspection
+ Sensible defaults works with auto default methods for baseline modeling
+ uses standard data flow of numpy structure as matrix
+ simplicity with simple interface with composible units

### Building blocks

1. Transformer

    transform() -> for transforming dataset
    fit() -> learns parameters | can induce data leakage
    fit_transform() -> learns and fit | using k-fold | better option

2. Estimator

    estimator element for estimating
    based on training dataset, and hyperparameter 
    fit() method only

3. Predictors

    Making predictio in dataset.
    predict() -> prediction through dataset
    score() -> score the prediction 

These component lined up into the sequence
data preprocessing -> training -> inference

### Dataset API endpoints

| endpoint | functionality | remarks |
| ------------- | -------------- | -------------- |
| sklearn.datasets | Loading datasets | custom dataset could also be loaded |
| sklearn.preprocessing | scaling, centering, normalization, binarization | essential preprocessing components |
| skelearn.impute | filling missing values | fills up missing values |
| sklearn.feature_selection | implements feature selection | feature selection layer component |
| skelearn.feature_extraction | helping into extraction  | usefull element into extracting features| 

