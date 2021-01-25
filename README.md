# Tide-transaction-receipt_Matching

-To run evaluation run following command :


```python src/evaluate.py --data_path XX --outfile_location result.csv ```

for evaluation you need to install Python Packages in the `requirenmnets.txt` file
-To Train the model run below command:

```python src/train.py --data_path XX --gpu 0```

Also for training you need to install Python Packages requirenmnets given in `train_requirenmnets.txt` file, or run the report notebook in a kaggle enviornment

-This project uses RapidsAI cuml, cudf to train XGBoost models on GPU, and Ray-Tune for HyperParameter Search

-But you can use pre-trained model just with xgboost installed for Evaluation

The Report contains description about model and data, in [`Report.ipynb`](https://github.com/Nilanshrajput/Tide-transaction-receipt_Matching/blob/master/Report.ipynb) file

`Artificats folder contains pre-trained model and data along with model configs file