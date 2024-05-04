import pandas as pd
import pickle
import xgboost
import shap
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter
import os

def make_classifier(csv_path="data_feature.csv")->None:
    directory = os.getcwd()+"/"
    csv_path = directory+csv_path
    save_path_model = directory+"pretrained_tools/pretrained_xgboost.pkl"
    save_path_x_train = directory+"pretrained_tools/X_train.pkl"
    df = output_df(csv_path=csv_path)
    X, y = df.drop(labels=['frame','cls'],axis=1), df['cls']
    #resample to boost tracking failure samples
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    model = xgboost.XGBClassifier().fit(X, y)
    with open(save_path_model, 'wb') as f:
        pickle.dump(model, f)
    with open(save_path_x_train, 'wb') as f:
        pickle.dump(X, f)

def output_df(csv_path:str)->pd.DataFrame:
    df = pd.read_csv(csv_path)
    occlusion_list = []
    for frame in df['frame'].unique():
        #for each frame
        new_df = df[df['frame']==frame].copy()
        curr_list = [0]*len(new_df)
        if len(new_df)==1:#if this frame just have a single box, there cannot be occlusion
            occlusion_list.extend(curr_list)
            break
        #when there are multiple boxes:
        for i in range(0,len(new_df)-1):
            for j in range(i+1,len(new_df)):
                if i!=j:
                    if if_occlusion(df=new_df,i=i,j=j):
                        curr_list[i] += 1
                        curr_list[j] += 1
        occlusion_list.extend(curr_list)
    df['occlusion'] = occlusion_list
    return df

def if_occlusion(df: pd.DataFrame,i:int,j:int)->bool:
    x_min_1 = df.iloc[i]['xmin']
    x_min_2 = df.iloc[j]['xmin']
    x_max_1 = df.iloc[i]['xmax']
    x_max_2 = df.iloc[j]['xmax']
    y_min_1 = df.iloc[i]['xmin']
    y_min_2 = df.iloc[j]['ymin']
    y_max_1 = df.iloc[i]['ymax']
    y_max_2 = df.iloc[j]['ymax']
    if (x_min_2>x_min_1 and x_min_2<x_max_1) or (x_max_2<x_max_1 and x_max_2>x_min_1):
        #when j's left side is inbetween i's width or when j's right side is inbetween i's width
        if (y_min_2<y_max_1 and y_min_2>y_min_1) or (y_max_2>y_min_1 and y_max_2<y_max_1):
            #when j's up side is inbetween i's height or when j's down side is inbetween i's height
            return True
    return False

make_classifier()