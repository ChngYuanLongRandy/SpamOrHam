import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 

def add_scamy_words(data:pd.DataFrame, scamy_words:List[str]):
    data['scamy_words']= data['text'].apply(lambda x: any([k in x for k in scamy_words]))
    data['scamy_words'] = data['scamy_words'].astype('int')

def add_length(data:pd.DataFrame):
    data['length'] = data['text'].apply(lambda x : len(x.strip()))
    empty_indicies = data[data['length']==0].index
    data.drop(index=empty_indicies, inplace= True)

def add_numbers(data:pd.DataFrame):
    data['numbers']= data['text'].str.extract(r"(\d+)")
    data[data.numbers.isna() == False]['numbers'] = 1
    data.numbers.fillna(0, inplace=True)
    data['numbers'] = data['numbers'].apply(lambda x: 1 if x !=0 else 0)

def add_features(data:pd.DataFrame,scamy_words:List[str]):
    add_length(data)
    add_scamy_words(data, scamy_words)
    add_numbers(data)

def remove_empy_strings(data_copy:pd.DataFrame):
    data = data_copy.copy()
    empty_indicies = data[data['length']==0].index
    data.drop(index=empty_indicies, inplace= True)
    return data

def transform(data:pd.DataFrame,scamy_words:List[str], training:bool= True):
    # remove dups
    data.drop_duplicates(inplace=True)

    # remove NAs
    data.dropna(inplace=True)

    # add features
    add_features(data,scamy_words)

    return data

def scale_after_transform(data:pd.DataFrame):
    pass

class Datapipeline:

    def __init__(
        self, 
        train_df:pd.DataFrame, 
        val_df:pd.DataFrame, 
        test_df:pd.DataFrame, 
        scamy_words: List[str] = ['call','text','won','now','free'], 
        countVec:bool = True, 
        tfidf:bool = False,
        return_text:bool = False
        ) -> None:
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.ss = StandardScaler()
        self.countvec = CountVectorizer()
        self.tfidf = TfidfVectorizer()
        self.countvec_bool = countVec
        self.tfidf_bool  = tfidf
        self.scamy_words = scamy_words
        self.return_text = return_text

        if (self.countvec_bool == True) and  (self.tfidf_bool == True):
            raise ValueError('Either countvec or tfidf should be True or False not True for both')


    def _add_scamy_words(self,data_copy:pd.DataFrame, scamy_words:List[str]):
        data = data_copy.copy()
        data['scamy_words']= data['text'].apply(lambda x: any([k in x for k in scamy_words]))
        data['scamy_words'] = data['scamy_words'].astype('int')
        return data

    def _add_length(self,data_copy:pd.DataFrame):
        data = data_copy.copy()
        data['length'] = data['text'].apply(lambda x : len(x.strip()))
        data = self._remove_empy_strings(data)
        return data

    def _remove_empy_strings(self,data_copy:pd.DataFrame):
        data = data_copy.copy()
        empty_indicies = data[data['length']==0].index
        data.drop(index=empty_indicies, inplace= True)
        return data

    def _add_numbers(self,data_copy:pd.DataFrame):
        data = data_copy.copy()
        data['numbers']= data['text'].str.extract(r"(\d+)")
        data[data['numbers'].isna() == False]['numbers'] = 1
        data['numbers'].fillna(0, inplace=True)
        data['numbers'] = data['numbers'].apply(lambda x: 1 if x !=0 else 0)
        return data


    def _add_features(self,data:pd.DataFrame,scamy_words:List[str]):
        data = self._add_length(data)
        data = self._add_scamy_words(data, scamy_words)
        data = self._add_numbers(data)
        return data

    def _fit(self, df:pd.DataFrame):
        self.ss.fit(df['length'].values.reshape(-1,1))
        
        if self.countvec_bool:
            self.countvec.fit(df['text'])
        else:
            self.tfidf.fit(df['text'])


    def transform(self,df:pd.DataFrame):
        # remove dups
        df.drop_duplicates(inplace=True)

        # remove NAs
        df.dropna(inplace=True)

        # add features
        df = self._add_features(df,self.scamy_words)
        df_train = self._add_features(self.train_df, self.scamy_words)

        self._fit(df_train)

        y = df.pop('spam')

        df['length'] = self.ss.transform(df['length'].values.reshape(-1,1))
        if self.countvec_bool:
            df_text = self.countvec.transform(df['text'])
            if self.return_text == False:
                df.drop(columns=['text'], inplace=True)
            X = np.concatenate([df_text.toarray(),df.values], axis=1)
        elif self.tfidf_bool:
            df_text = self.tfidf.transform(df['text'])
            if self.return_text == False:
                df.drop(columns=['text'], inplace=True)
            X = np.concatenate([df_text.toarray(),df.values], axis=1)
        else:
            if self.return_text == False:
                df.drop(columns=['text'], inplace=True)
            X = df.values

        if self.return_text == False:
            return X, y
        else:
            text_df = df['text']
            df.drop(columns=['text'], inplace=True)
            X = df.values
            return text_df , X, y