import numpy as np
import pandas as pd
from collections import Counter
# from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
# from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm
def extract_txt_file(path):
    with open(path, 'r') as file:
        report = file.read()

    report = report.replace('___', '')
    report = report.replace('\n', '')
    return report
def mergeMIMIC():

    result = pd.read_csv('/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv')
    traintest_splits = pd.read_csv('/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv')
    df = pd.read_csv('/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv')
    df = df.loc[(df['ViewPosition'] == 'PA') | (df['ViewPosition'] == 'AP')]
    new_result = pd.DataFrame(columns=np.append(result.columns.values, np.array(['Path', 'Path_compr', 'Path_text', 'Report', 'split'])), index=range(df["dicom_id"].values.shape[0]))
    print(new_result)
    paths = df["dicom_id"].values.copy()
    empty = 0
    c_nf = 0
    for i in tqdm(range(paths.shape[0])):
        p_compr = '/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/compressed_images224/files/' + 'p{}/p{}/s{}/'.format(
            str(df['subject_id'].values[i])[:2], df['subject_id'].values[i], df['study_id'].values[i]) + paths[
                     i] + '.jpg'
        p_txt = '/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr/2.0.0/files/' + 'p{}/p{}/s{}.txt'.format(
            str(df['subject_id'].values[i])[:2], df['subject_id'].values[i], df['study_id'].values[i])

        p = '/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'+'p{}/p{}/s{}/'.format(str(df['subject_id'].values[i])[:2], df['subject_id'].values[i],df['study_id'].values[i])+paths[i]+'.jpg'

        result_index = result.index[(result['subject_id'] == df['subject_id'].values[i]) & (result['study_id'] == df['study_id'].values[i])]
        split_index = traintest_splits.index[traintest_splits['dicom_id'] == paths[i]].tolist()
        split = str(traintest_splits.loc[split_index[0]]['split'])
        try:
            class_values = result.loc[result_index].values[0]
            class_values = np.nan_to_num(class_values)
            class_values = np.where(class_values==-1.0, 0.0, class_values)
            class_values = np.where(class_values == -9.0, 0.0, class_values)
            report = extract_txt_file(p_txt)
            if np.count_nonzero(class_values[2:])==0:
                class_values[10] = 1.0

            row = list(class_values) + [p, p_compr, p_txt, report, split]

            new_result.iloc[i] = row
            c_nf+=1

        except:
            print("Error preprocessing data")
    new_result.to_pickle("/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/train_test_val_data_1904.pkl")


def stratify(in1,out1,out2):
    df = pd.read_pickle(in1)
    df_train = df.loc[df['split']=='train'].reset_index()
    df_test= df.loc[df['split']=='test'].reset_index()

    print("Train Samples: {}".format(len(df_train.index)))
    print("Test Samples: {}".format(len(df_test.index)))
    df_train.to_pickle(out1)
    df_test.to_pickle(out2)
                    
                    
def stratify_val(in1,out1,out2):
    df = pd.read_pickle(in1)
    print(df.head())
    train = df.sample(frac=0.8,random_state=200) #random state is a seed value
    val  = df.drop(train.index)
    print("Train Samples: {}".format(len(train.index)))
    print("Val Samples: {}".format(len(val.index)))

    train.to_pickle(out1)
    val.to_pickle(out2)
if __name__ == '__main__':

    # mergeMIMIC()
    stratify_val("/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/improved_multi_mimic_0605_2_fullehr224_BioBERT.pkl",
             "/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/vkd_train2_1309.pkl",
             "/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/vkd_test_1309.pkl")
    stratify_val("/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/vkd_train2_1309.pkl",
             "/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/vkd_train_1309.pkl",
             "/media/tjvsonsbeek/Data1/physionet.org/files/mimic-cxr-jpg/2.0.0/vkd_val_1309.pkl")
