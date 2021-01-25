import cupy as cp # linear algebra
import cudf # data processing, CSV file I/O (e.g. cudf.read_csv)
from cuml.preprocessing.model_selection import train_test_split
from cuml.preprocessing.LabelEncoder import LabelEncoder
from cuml.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split as sktrain_test_split
import sklearn



from random import shuffle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost
from cupy import asnumpy


from ray.tune.schedulers import ASHAScheduler
from ray import tune
from ray.tune.integration.xgboost import TuneReportCheckpointCallback

import os
from argparse import ArgumentParser


def prep_data(path):
    """ Read and Prepare data for training """

    data = cudf.read_csv(path,sep=":")

    # Create Label column
    data['label']= (data.matched_transaction_id == data.feature_transaction_id).astype(int)

    # Create features and target for training      
    x,y = data[list(data.columns[4:])], data.label
    x=x.drop(['label'], axis=1)
    
    return x,y

def evaluate_results(model, test):
    """ Evaluate model on test data, generate metrics scores """
    preds = model.predict(test)
    print(classification_report(asnumpy(test),preds.round(0)))
    print( " In the Confusion Matrix below, the digonal values represent correct classification for each class : ")
    labels = ['label-0', 'label-1']
    
    # Confusion matrix
    cm = sklearn.metrics.confusion_matrix(asnumpy(test),asnumpy(preds.round(0).astype(int)))

    # Plot the confusion matrix
    ax= plt.subplot()
    sns.heatmap(cm.astype(int), annot=True,fmt='g', ax = ax); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);
            

def train_receipt_match(config, hyp_search=True):
    """ Traning function """

    # Read and prepare the training data for training
    x,y = prep_data(path)

    # Divide the data for training and testing 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42,shuffle=True)
    train = xgboost.DMatrix(X_train, label=y_train)
    test = xgboost.DMatrix(X_test, label=y_test)

    # The data to visualise model performance  during training phase
    watchlist = [(test, 'eval'), (train, 'train')]

    if hyp_search:
        clf = xgboost.train(config, train, num_boost_round=20000,
                            evals=watchlist,
                            maximize=True,
                            verbose_eval=1000,
                            callbacks=[TuneReportCheckpointCallback(filename="model.xgb")]
                        )

        # evaluate model on test data
        evaluate_results(clf, test)
    else:
        clf = xgboost.train(config, train, num_boost_round=20000,
                        evals=watchlist,
                        maximize=True,
                        verbose_eval=1000,
                    )

        # evaluate model on test data
        evaluate_results(clf, test)
    

   
        
if __name__ == "__main__":
    parser = ArgumentParser(description="Tide Receipt matching XGB classifier")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../artifacts/train_Data.csv",
        help="Train data csv file path",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to be used for parameter search",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="1 is GPU is available or 0",
    )

    parser.add_argument(
        "--hyper_paramsearch",
        type=bool,
        default=False,
        help="True if you want to use hyperparmeter search, false to use pre-set best parameter",
    )

    args = parser.parse_args()
    path = args.data_path
    use_gpu = args.gpu 
    search_samples = args.num_samples
    hyp_search = args.hyper_paramsearch
    hardware = "gpu" if use_gpu else "cpu"

    if hyp_search:

        # Search Space for the hyperparameters
        search_space = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "error", "aucpr"],
            "max_depth": tune.randint(1, 9),
            "min_child_weight": tune.choice([1, 2, 3]),
            "subsample": tune.uniform(0.4, 1.0),
            "eta": tune.loguniform(1e-4, 1e-1),
            "scale_pos_weight":tune.randint(3, 15),
            "colsample_bytree": tune.uniform(0.1, 1)
        }
        
        # This will enable aggressive early stopping of bad trials.
        scheduler = ASHAScheduler(
            max_t=10,  # 10 training iterations
            grace_period=1,
            reduction_factor=2)

        # Start HyperParameter search
        analysis = tune.run(
            train_receipt_match,
            metric="eval-aucpr",
            mode="max",
            # You can add "gpu": 1 to allocate GPUs
            resources_per_trial={"hardware": 1},
            config=search_space,
            num_samples=20,
            scheduler=scheduler)
    
    else:
        xgb_params = {
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist',
            'max_depth': 2, 
            'eta':0.1,
            'silent':1,
            'subsample':0.5,
            'colsample_bytree': 0.05,
            'scale_pos_weight':8
            }
        train_receipt_match(config=xgb_params, hyp_search=False)