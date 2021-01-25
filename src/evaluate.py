from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split as sktrain_test_split
import sklearn
from random import shuffle
import numpy as np 
import pandas as pd 
import os
from argparse import ArgumentParser

import xgboost

def process_data(data):
    x = data[list(data.columns[4:])]
    x = xgboost.DMatrix(x)
    return x

def generate_output(preds, output_file, df):
    """ Generates output csv in decreasing order of probabilitiy of match"""
    out_df = df[df.columns[0:4]]
    out_df['scores'] = preds
    out_df.sort_values('scores', ascending= False, inplace = True)
    out_df.to_csv(output_file)


if __name__ == "__main__":
    parser = ArgumentParser(description="Tide Receipt matching XGB classifier")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to test csv data",
    )
    parser.add_argument(
        "--outfile_location",
        type=str,
        default="result.csv",
        help="The location to save results",
    )

    # Parse the cli argumnets
    args = parser.parse_args()
    path = args.data_path
    output_file = args.outfile_location

    # Create xgb model
    model = xgboost.Booster()

    # Load pre-trained model
    model_path = "../artifacts/model.xgb"
    model.load_model(model_path)

    # Read the test data 
    df = pd.read_csv(path, sep=":")
    x = process_data(df)

    # Generate Match probability scores
    preds=model.predict(x)

    # Saves the result in the decereasing order of likelihood of matching a receipt image
    generate_output(preds,output_file, df)

    