import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset


def get_df(data_dir, train_step):

    df = pd.read_csv('train_0.csv')

    if train_step == 0:
        df_train = pd.read_csv(os.path.join(data_dir, 'train.csv')).drop(columns=['url'])
    else:
        cls_81313 = df.landmark_id.unique()
        df_train = pd.read_csv(os.path.join(data_dir, 'train.csv')).drop(columns=['url']).set_index('landmark_id').loc[cls_81313].reset_index()
    
    # delete not existing pathfiles
    print('*********************** deleting not existing pathfiles ***********************')

    df_train['filepath_flag'] = df_train['id'].apply(lambda x: os.path.exists(os.path.join(data_dir, 'train', x[0], x[1], x[2], f'{x}.jpg')))
    delete = df_train[df_train['filepath_flag'] == False]
    df_train = df_train.drop(delete.index)
    
    print('*********************** deleting finished ***********************')

    df_train['filepath'] = df_train['id'].apply(lambda x: os.path.join(data_dir, 'train', x[0], x[1], x[2], f'{x}.jpg'))

    df = df_train.merge(df, on=['id','landmark_id'], how='left')

    landmark_id2idx = {landmark_id: idx for idx, landmark_id in enumerate(sorted(df['landmark_id'].unique()))}
    idx2landmark_id = {idx: landmark_id for idx, landmark_id in enumerate(sorted(df['landmark_id'].unique()))}
    df['landmark_id'] = df['landmark_id'].map(landmark_id2idx)

    print('*********************** save to test.csv ***********************')

    df.to_csv('test.csv')

    print('*********************** save finished ***********************')


if __name__ == '__main__':

    data_dir = './data'
    train_step = 0

    get_df(data_dir, train_step)