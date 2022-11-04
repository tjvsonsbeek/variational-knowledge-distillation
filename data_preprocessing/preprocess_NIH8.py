import pandas as pd
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import save_img
import os.path
import errno
from tqdm import tqdm
import numpy as np
import copy

from collections import Counter
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
from skmultilearn.model_selection import iterative_train_test_split


def open_file(filename):
    with open(filename) as f:
        content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    return content
def resize_images(filename, target_size):
    # resize images to new target size. Input is pandas dataframe of full dataset before splitting of train/val/test
    df = pd.read_csv(filename)
    rows_to_remove_train = []
    print(df.head())
    rows_to_remove_test = []
    train_files = open_file('../train_val_list.txt')
    test_files = open_file('../test_list.txt')

    classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices']
    classes_i = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices', 'Img']
    dirs = ['images_001', 'images_002', 'images_003', 'images_004', 'images_005', 'images_006', 'images_007', 'images_008', 'images_009', 'images_010', 'images_011', 'images_012']
    classes_nih = [1,0,1,0,0,1,1,1,1,1,0,0,0,0]
    df = df.loc[(df['View Position'] == 'PA') | (df['View Position'] == 'AP')]
    new_df_train = pd.DataFrame(columns=np.array(classes_i), index=range(df["Image Index"].values.shape[0]))
    print(new_df_train.head())
    new_df_test = pd.DataFrame(columns=np.array(classes_i), index=range(df["Image Index"].values.shape[0]))
    path = copy.deepcopy(df['Image Index'].values)
    
    path_compr= copy.deepcopy(path)
    for idx in tqdm(range(len(path))):

        if path[idx] in train_files:
            rows_to_remove_test.append(idx)
        elif path[idx] in test_files:
            rows_to_remove_train.append(idx)
            ##
            nz = False
            for i in range(14):
                if classes[i] in df['Finding Labels'][idx]:
                    new_df_train[classes[i]][idx] = 1
                    new_df_test[classes[i]][idx] = 1
                    nz = True
                else:
                    new_df_train[classes[i]][idx] = 0
                    new_df_test[classes[i]][idx] = 0
                    nz =True
            if not nz:
                rows_to_remove_train.append(idx)
                rows_to_remove_test.append(idx)
            else:
                path_compr[idx] = path_compr[idx][:-4] + '_224' + path_compr[idx][-4:]
                sw = True

                for d in dirs:
                    if path[idx] in os.listdir('../'+d+'/images/'):
                        path[idx] = '../'+d+'/images/' + path[idx]
                        path_compr[idx] = '../'+d+'/images/' + path_compr[idx]
                        sw = False
    new_df_train['Img'] = path_compr
    new_df_train=new_df_train.drop(new_df_train.index[rows_to_remove_train])
    new_df_test['Img'] = path_compr
    new_df_test=new_df_test.drop(new_df_test.index[rows_to_remove_test])
    print(new_df_test.head())
    print(new_df_train.head())
    # save new version in which instances without valid image are removed
    new_df_train.to_pickle(filename[:-4]+'train_intermediate.pkl')
    new_df_test.to_pickle(filename[:-4]+'test.pkl')

def stratify_val(name_train_val, name_train, name_val):
    df = pd.read_pickle(name_train_val)

    train = df.sample(frac=0.8,random_state=200) #random state is a seed value
    val  = df.drop(train.index)
    print("Train Samples: {}".format(len(train.index)))
    print("Val Samples: {}".format(len(val.index)))

    train.to_pickle(name_train)
    val.to_pickle(name_val)

if __name__ == '__main__':
    resize_images("../Data_Entry_2017.csv", target_size = (224,224))
    stratify_val('../train_intermediate.pkl', '../val.pkl', '../train.pkl')

