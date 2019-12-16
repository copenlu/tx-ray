########################################################################################################################
"""
Description :  Takes the raw data and compiles it in the form of a csv file for the supervised case.
Python version : 3.5.3
author Vageesh Saxena
"""
########################################################################################################################


################################################ Importing libraries ###################################################
import os
import re

import pandas as pd
########################################################################################################################


def convert_raw_data_into_csv():
    # Getting the names of all the raw files
    train_pos_files = os.listdir("data/aclImdb/train/pos/")
    train_neg_files = os.listdir("data/aclImdb/train/neg/")
    test_pos_files = os.listdir("data/aclImdb/test/pos/")
    test_neg_files = os.listdir("data/aclImdb/test/neg/")

    para, sentiment, datatype = ([] for i in range(3))
    for file in train_pos_files:
        with open(os.path.join("data/aclImdb/train/pos/", file)) as f:
            for line in f:
                para.append(re.sub('[\[\]/{}]+', '', line).replace("<br >",""))
                sentiment.append("pos")
                datatype.append("train")
                
    for file in train_neg_files:
        with open(os.path.join("data/aclImdb/train/neg/", file)) as f:
            for line in f:
                para.append(re.sub('[\[\]/{}]+', '', line).replace("<br >",""))
                sentiment.append("neg")
                datatype.append("train")
                
    for file in test_pos_files:
        with open(os.path.join("data/aclImdb/test/pos/", file)) as f:
            for line in f:
                para.append(re.sub('[\[\]/{}]+', '', line).replace("<br >",""))
                sentiment.append("pos")
                datatype.append("test")
                
    for file in test_neg_files:
        with open(os.path.join("data/aclImdb/test/neg/", file)) as f:
            for line in f:
                para.append(re.sub('[\[\]/{}]+', '', line).replace("<br >",""))
                sentiment.append("neg")
                datatype.append("test")

    # Saving data to a csv file named as imdb_master.csv
    df = pd.DataFrame(columns=["review", "type", "label"])
    df["review"] = para
    df["type"] = datatype
    df["label"] = sentiment

    df.to_csv(os.path.join("data","imdb_master.csv"), index=False)
