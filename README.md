# K-Nearest Neighbors classification algorithm

## Algorithm
> __Distance__ `Euclidean`

> __Classifier__ `Majority Vote | Weighted Distance`

> __Feature Normalization__ `None`

> __Dimension Reduction__ `None`

> __Cross-Validation__ `None`


## Assumptions
* File format `csv`
* Input data `shuffled`
* Number of features `< 10`
* Feature units `scaled`
* Attribute datatype `real`
* Missing values `false`
* Headers `false`

## Configuration
```
{
    "input": <path: string>,
    "exclude": <columns: integer[]>,
    "class": <column: integer>,
    "neighbors": <count: integer>,
    "skewed": <true/false: boolean>,
    "sample": <size: integer>,
    "test": <percentage: float>
}
```
#### Example
```
{
    "input": "./skin.csv",
    "exclude": [],
    "class": 3,
    "neighbors": 5,
    "skewed": true,
    "sample": 10000,
    "test": 0.2
}
```
