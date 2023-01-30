import pickle
import pandas as pd
from api.api.config import SETTINGS
from typing import List
from sklearn.model_selection import train_test_split
from api.src.datapipeline import Datapipeline

def load_model(model_path:str):
    with open(model_path, 'rb')as f:
        MODEL = pickle.load(f)
    return MODEL

def load_dataset(file_path:str):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)
    df.columns = ['spam','text']
    df.spam = df.spam.apply(lambda x: 0 if x == 'ham' else 1)
    return df

def return_train_val_test_sets(file_path:str, seed:int = SETTINGS.SEED, \
    test_size:float=SETTINGS.TEST_SIZE, stratify:bool=SETTINGS.STRATIFY)->List[pd.DataFrame]:
    """returns the train val test set, making the columns correct."""
    df = load_dataset(file_path)

    if stratify:
        train_df, val_df = train_test_split(df,test_size=test_size,random_state=seed, stratify=df['spam'])
        val_df, test_df = train_test_split(val_df,test_size=0.5,random_state=seed, stratify=val_df['spam'])

        return train_df, val_df, test_df
    else:
        train_df, val_df = train_test_split(df,test_size=test_size,random_state=seed)
        val_df, test_df = train_test_split(val_df,test_size=0.5,random_state=seed)

        return train_df, val_df, test_df

def prep_pipeline(file_path:str):
    train_df, val_df, test_df = return_train_val_test_sets(file_path, SETTINGS.SEED, SETTINGS.TEST_SIZE, SETTINGS.STRATIFY)
    pipe = Datapipeline(train_df, val_df, test_df)
    _, _ = pipe.transform(train_df)

    return pipe

PRED_MODEL = load_model(SETTINGS.MODEL_PATH)
DATASET = load_dataset(SETTINGS.DATA_PATH)
PIPELINE = prep_pipeline(SETTINGS.DATA_PATH)

