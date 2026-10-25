# Chaining transformers

With sklearn base components, we can compose multiple transformers to be used
as a single pipeline component, for consistent interface for passing training, 
and testing pipelines.
chain of transformer and estimators 

1. Pipeline

    Construct a chain of multiple transformer,
    execute a fixed sequence of steps with dataset, 
    preprocessing and modeling

    Works to load up transformers

    `sklearn.pipeline.Pipeline` interface

    sequentially apply transformers and estimators, 
    intermediate steps must apply fit and transform

    Assembeled steps into one for easy validation

    Building pipeline with `Pipeline()`
    input set list of tuples fed into the pipeline constructor
    
    benifits with pipeline
    simple one step training steps
    `pipe.fit_transform(dataset)`
    calling all transformers and estimator in one shot
    easy usage in testing and validation sequence

    **GridSearchcv** | Makes simplicity for parameter and combination selection

    We can also specify to store the transformation due to their expensive nature
    `Pipeline(steps=steping_stones, memory='pathway')` storing the transforming parametrs and finding

2. FeatureUnion

    Combining output of several transformation into one
    transformation feature,

    Loading estimators 
    with feature union we can compose more rich steping with transformations, 
    estimator to the dataflow with simple composable component based code.
