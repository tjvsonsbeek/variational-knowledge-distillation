import pandas as pd
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import save_img
import os.path
import errno
from tqdm import tqdm
import numpy as np
import copy
from BERTtokenizer import BiobertEmbedding
def resize_images(filename, path_processed, target_size):
    # resize images to new target size. Input is pandas dataframe of full dataset before splitting of train/val/test
    df = pd.read_pickle(filename)
    bert = BiobertEmbedding()
    path = df['Path'].values
    path_compr = df['Path_compr'].values
    path_text = copy.deepcopy(path_compr)
    rows_to_remove = []
    for idx in tqdm(range(len(path))):
        try:
            # if path[idx][0] != '/':
            #     path[idx] = '/' + path[idx]
            #     path_compr[idx] = '/' + path_compr[idx]
            #     path_text[idx] = '/' + path_text[idx]
            # path_compr[idx] = path_compr[idx].replace('224','512')
            
            path_compr[idx] = path_compr[idx]
            path_text[idx] = path_text[idx][:-4]+'bioberttext.png'
            # if not os.path.exists(path_compr[idx]):
            #     try:
            #         os.makedirs(os.path.dirname(path_compr[idx]), exist_ok=True)
            #     except OSError as exc:  # Guard against race condition
            #         if exc.errno != errno.EEXIST:
            #             print("race condition")
            #             raise
            
            # if not os.path.exists(path_compr[idx]):
            array = img_to_array(load_img(path[idx], target_size = target_size))
            save_img(path_compr[idx], array)
            
            # if not os.path.exists(path_text[idx]):
            # print(df['Report'][idx])
            embed = np.expand_dims(bert.word_vector(df['Report'][idx]),2)
            save_img(path_text[idx], embed)
        except:
            rows_to_remove.append(idx)
            print('error in processing')
        
    df['Path_compr'] = path_compr
    df['Path'] = path
    df['Path_text'] = path_text
    df=df.drop(df.index[rows_to_remove])
    # save new version in which instances without valid image are removed
    df.to_pickle(path_processed)

if __name__ == '__main__':
    path_unprocessed = "/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/train_test_val_data_1904.pkl"
    path_processed = "/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/train_test_val_data_1904_improved.pkl"
    resize_images(path_unprocessed, path_processed, target_size = (224,224))
