# Tide-transaction-receipt_Matching

### Running the code

-To run evaluation run following command :


```python src/evaluate.py --data_path XX  ```

This will genreate the results for ‘transaction-receipt’ match with decreasing order of likelihood of matching a receipt.

The parameters can be overridden via the command line:

1. --data_path - Path to test csv data.
2. --outfile_location - The location to save results.


For evaluation you need to install Python Packages in the `requirenmnets.txt` file.


-To Train the model run below command:

```python src/train.py ```

This will run the training on cpu and data source will from repostiory only.

The parameters can be overridden via the command line:

1. --data_path - Train data csv file path.
2. --gpu - 1 is GPU is available or 0
3. --hyper_paramsearch - True if you want to use hyperparmeter search, false to use pre-set best parameter.

Also for training you need to install Python Packages requirenmnets given in `train_requirenmnets.txt` file, or run the report notebook in a kaggle enviornment.

-This project uses RapidsAI cuml, cudf to train XGBoost models on GPU, and Ray-Tune for HyperParameter Search.

-But you can use pre-trained model just with xgboost installed for Evaluation.

## Report

- Link to the Kaggle Kernel To see or run the Report https://www.kaggle.com/nilanshrajput/report 
- Report contains information about the data and Modeling process, also you can train and test results firectly from report in kaggle kernel.

[`Artifcats`](https://github.com/Nilanshrajput/Tide-transaction-receipt_Matching/tree/master/artifacts) folder contains pre-trained model and data along with model configs file.



## References 

- https://docs.ray.io/en/master/tune/tutorials/tune-xgboost.html
- https://medium.com/rapids-ai/financial-data-modeling-with-rapids-5bca466f348
