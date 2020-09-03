import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import os
import glob
import json
import sys
import gzip
from helpers import create_folder_path, write_csv_file, write_list_to_txt, save_pk_file

def read_jsonl_file( file_name, columns):
    """
    Read jsonl file and create a data frame of the required data
    file_name: the name of the read file
    columns: a list of columns that need to be extracted, 
            if columns is None, all data from the jsonl file will be converted to dataframe
    sort_value: the value that is used to sort the dataframe, default value is "id"
    Return a dataframe of data required to be extracted
    """
    with open(file_name,'r', encoding = 'utf-8') as jfile:
        records = [json.loads(line) for line in jfile]
        df = pd.DataFrame.from_records(records)
        sorted_df = df.sort_values(by="id", ascending=True)
        if columns == None:
            return sorted_df
        else:
            cleaned_df = sorted_df[columns]
            return cleaned_df
        

def prepare_json_data(inst_file, truth_file, inst_columns = None , truth_columns = None):
    """
    Read the files from the corpus including an instance file and a truth file.
    inst_file: the path to the instance file
    truth_file: the path to the truth file
    inst_columns: a list of columns required to extracted from the instance file, default value is None meaning all data are needed
    truth_column: a list of columns required to extracted from the truth file, default value is None meaning all data are needed
    Return a dataframe that is the combination of data from the instance file and the truth file
    """
    inst_df = read_jsonl_file(inst_file, inst_columns)
    truth_df = read_jsonl_file(truth_file, truth_columns)
    merged_df = pd.merge(inst_df, truth_df, on = 'id')
    
    return merged_df


def split_json_data(df, folder, column = "truthScale"):
    """
    Split data into two subset according to the label
    dataframe: the original dataframe
    column: the name of the columns that contain labels
    folder: the path to the folder containing new data file
    Return the path to the new file
    """
    value_set = set(df[column])
    for value in value_set:
        splited_data = df[df[column]== value]
        headline = list(splited_data["targetTitle"])
        textbody = list(splited_data["targetParagraphs"])
        
        headline_file_path = f'{folder}/headline_{value}'
        textbody_file_fath = f'{folder}/textbody_{value}'
        
        write_list_to_txt(headline,headline_file_path)
        write_list_to_txt(textbody,textbody_file_fath)


def read_jsonl_folder(json_folder):
    """
    Read the instance.jsonl and truth.jsonl the folder
    json_folder: the path to the folder that contain the two files
    write_folder: the path to the folder that contain the outfile
    Return the name of the outfile
    """
    inst_columns = ['id',"targetTitle","targetParagraphs"]#, 'postMedia','postText']
    truth_columns = ["id","truthClass"]#, "truthMode","truthJudgments"]
    path_inst_file = json_folder+"/instances.jsonl"
    path_truth_file = json_folder+"/truth.jsonl"

    merged_df = prepare_json_data(path_inst_file, path_truth_file, inst_columns, truth_columns)

    merged_df["targetTitle"] = merged_df["targetTitle"].progress_map(lambda x: str(x).strip("[").strip(']').strip("\'").strip('\"'))
    #merged_df['postText'] = merged_df['postText'].progress_map(lambda x: ' '.join(map(str, x)))
    #merged_df['postMedia'] = merged_df['postMedia'].progress_map(lambda x: 0 if x == "[]" else 1)
    merged_df['targetParagraphs'] = merged_df['targetParagraphs'].progress_map(lambda x: ' '.join(map(str, x)))
    #merged_df["truthScale"] = merged_df["truthMode"].progress_map(lambda x: "non" if x == 0.0 else ("slightly" if 0.3<x<0.6 else ("considerable" if 0.6<x<1 else "heavy")))
    merged_df["truthClass"] = merged_df["truthClass"].progress_map(lambda x: "CB" if x == "clickbait" else "Non")

    drop_df = merged_df[~merged_df.targetTitle.str.contains("Sections Shows Live Yahoo!")]
    final_df = drop_df[~drop_df.targetTitle.str.contains("Top stories Top stories")]


    write_csv_file(final_df, json_folder)
    pk_file = save_pk_file(final_df, json_folder)
    #split_json_data(final_df, save_to)
    print(final_df[:3])

    return pk_file

def gz_to_txt(gz_file, txt_file):
    """
    Convert gz file to txt file and convert content format from byte to utf8
    gz_file: the path gz file that need to be converted
    txt_file: the path gz file that need to be converted
    Print a statement that file created
    """
    with gzip.open(gz_file, 'rb') as outfile:
        file_content = outfile.read()
        with open (txt_file,"w", encoding="utf8") as infile:
            infile.write(file_content.decode("utf-8"))
            print( "File {} created".format(txt_file))


def read_txt(file_name):
    """
    Read txt file and return a dataframe containing the data
    file_name: the name of txt file
    """
    with open (file_name, "r", encoding = "utf8") as infile:
        content = infile.readlines()
        df = pd.DataFrame()
        lines = []
        for line in content:
            if line != "\n":
                new_line = line.strip("\n")
                lines.append(new_line)
        df["targetTitle"] = lines
        df["truthClass"] = "Non" if "non" in file_name else "CB"
    
    return df

def read_gz_folder(gz_folder):
    """
    read .gz files and return a dataframe contain the data in the file
    gz_folder: path to folder containing .gz files
    """
    df_list = []
    for read_file in tqdm(glob.glob(os.path.join(gz_folder, '*.gz'))):
        file_name = read_file.replace(".gz", ".txt")
        gz_to_txt(read_file, file_name)
        df = read_txt(file_name)
        df_list.append(df)
    merged_df = pd.concat(df_list)

    write_csv_file(merged_df, gz_folder)
    pk_file = save_pk_file(merged_df, gz_folder)
    print(merged_df[:5])

    return pk_file

def read_data(read_folder, extention):
    if extention == "json":
        read_jsonl_folder(read_folder)
    elif extention == "gz":
        read_gz_folder(read_folder)
    else:
        pass

if __name__ == "__main__":
    #if  len(sys.argv) != 4:

    #    print("python read_data.py read_folder write_folder json/gz")
    #else:
    #    read_folder = sys.argv[1]
    #    write_folder = sys.argv[2]
    #    extention =  sys.argv[3]
    #    read_data(read_folder, write_folder, extention)
    directories = os.listdir("Data")
    print(directories)
    for directory in directories:
        if "." not in directory:
            if "clickbait17" in directory:
                read_data("Data/"+directory, "json")
            else:
                read_data("Data/"+directory, "gz")
    
    #read_data("clickbait17-validation-170630", "clickbait17-validation-170630", "json")
    #read_data("Dataset_Charabokty", "Data", "gz")