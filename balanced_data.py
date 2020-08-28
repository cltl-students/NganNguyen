import pandas as pd
import os
import glob
import helpers
from sklearn.utils import shuffle
from helpers import load_pk_file, save_pk_file, create_folder_path

if __name__ == "__main__":
    ## reduce the number of non-clickbait samples in the dataset by randomly select n non-clickbait sample with n is the number of clickbait samples
    folder = create_folder_path("Train")
    Potthast_corpus = []
    Chakraborty_corpus = []

    for file_name in glob.glob(os.path.join("Processed_data", '*.pk')):
        df = load_pk_file(file_name)
        cb = df.loc[df['truthClass'] == "CB"]
        non = df.loc[df['truthClass'] == "Non"]
        
        if "clickbait17" in file_name:
            Potthast_corpus.append(cb)
            Potthast_corpus.append(non.sample(n = len(cb)))
        else:
            Chakraborty_corpus.append(cb.sample(n = 5000))
            Chakraborty_corpus.append(non.sample(n = 5000))

    save_Potthast = pd.concat(Potthast_corpus)
    save_Chakraborty = pd.concat(Chakraborty_corpus)

    Potthast_data = shuffle(save_Potthast, random_state=5)
    Chakraborty_data = shuffle(save_Chakraborty, random_state=5)

    save_pk_file(Potthast_data, folder+"/Potthast_data")
    save_pk_file(Chakraborty_data, folder+"/Chakraborty_data")
