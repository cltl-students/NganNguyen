import os
import pandas as pd
import pickle

def create_folder_path(folder_path):
    """
    Create new folders for data seperation
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return folder_path
    else:
        return folder_path

def read_csv_file(file_name):
    """
    Read the file from path to file
    path: the path the reading file
    """
    df = pd.read_csv(file_name, sep='\t',encoding='utf-8')
    return df

def write_csv_file(dataframe, path):
    """
    Write new file with provided dataframe
    path: the path to the new file
    dataframe: the data that need to be saved
    """
    file_name = path+".csv"
    dataframe.to_csv(file_name, sep='\t',  encoding='utf-8')
    print( "File {}.csv created".format(path))
    return file_name

#def read_df_column(dataframe, column_name = "id"):
#    column = dataframe[column_name]
#    return column

def save_pk_file(data, path):
    """
    Save data to a file using Pickle
    path: the path to the save file
    Return: file name
    """
    file_name = path+".pk"
    with open(file_name,'wb') as f:
        pickle.dump(data, f)
    print( "File {}.pk created".format(path))
    return file_name


def load_pk_file(file_name):
    """
    Load data from a file using Pickle
    path: the path to the required file
    Return: the data in the form before saved
    """
    with open(file_name,'rb') as f:
        data = pickle.load(f)
    return data

def merge_list_of_lists(list_of_lists):
    """
    Merge a list of lists into one flat list
    list_of_lists: a list contains many lists as items
    Return merged a list
    """
    merged_list = sum(list_of_lists, [])
    return merged_list

def write_list_to_txt(list_name, path):
    file_name = path+".txt"
    with open (file_name, "w", encoding="utf8") as f:
        for item in list_name:
            f.write(str(item))
            f.write("\n")
    
    print("File {}.txt created".format(file_name))
    return file_name
