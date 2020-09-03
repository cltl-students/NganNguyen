import numpy as np
import pandas as pd
from text_process import Text_process
from helpers import create_folder_path,load_pk_file,save_pk_file
from sklearn.feature_extraction import DictVectorizer
from sklearn import utils
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import multiprocessing
import spacy
nlp = spacy.load("en_core_web_sm")
import os
import glob
from tqdm import tqdm
tqdm.pandas()

def train_d2v(documents, model_folder):
    """
    Train Doc2Vec model using CBOW 
    documents: a list of tokenised documents for training
    model_folder: folder to save trained model
    Return the path to saved model
    """
    print("Training 100 dimension document embedding model")

    tagged_documents = [TaggedDocument(doc,[i]) for i, doc in enumerate(documents)]
    cores = multiprocessing.cpu_count()
    model_dbow = Doc2Vec(dm=1,vector_size=100,negative=5,min_count=2,workers=cores,alpha = 0.025,min_alpha=0.025)
    model_dbow.build_vocab(tagged_documents)
    
    for epoch in tqdm(range(10)):
        model_dbow.train(utils.shuffle(tagged_documents), total_examples=model_dbow.corpus_count, epochs=epoch)
        model_dbow.alpha -= 0.0002
        model_dbow.min_alpha = model_dbow.alpha

    
    d2v_model_file = model_folder+"/d2v_model"
    model_dbow.save(d2v_model_file)
    model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    print("Training doc2vec completed")

    return d2v_model_file

def train_w2v(tokenised_texts,model_folder, size=100):
    """
    Train word embeddings using Word2Vec model from Gensim library
    This function has one positional argument "tweets" and one key word function "size"
    tokenised_texts: a list of tokenised texts from the data file
    model_folder:path to folder contain saved model
    size: the size of the embedding, default value is 100
    Return the path to model file
    
    References:
    Documentation for Word2Vec model in Gensim library
    https://radimrehurek.com/gensim/models/word2vec.html
    """
    print(f"Training {size} dimension word embedding model")
    #Training model
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(size=size, window=5,min_count=1,workers=cores,sg=0,hs=1,cbow_mean=1,min_alpha=0.025)
    w2v_model.build_vocab(tokenised_texts)
    for epoch in tqdm(range(10)):
        w2v_model.train(utils.shuffle(tokenised_texts), total_examples=w2v_model.corpus_count, epochs=epoch)
        w2v_model.alpha -= 0.0002
        w2v_model.min_alpha = w2v_model.alpha

    #w2v_model.train(tokenised_texts, total_examples=w2v_model.corpus_count,epochs=model.epochs)
    w2v_model_file =model_folder+ "/w2v_model"
    w2v_model.save(w2v_model_file)
    
    print("Training word2vec completed")
    return w2v_model_file

    
def train_model(in_folder):
    model_folder = create_folder_path("Model")
    texts = []
    #contents = []

    for file_name in glob.glob(os.path.join(in_folder, '*.pk')):
        print(f"Reading {file_name}")
        df = load_pk_file(file_name)
        texts.extend(list(df["token"]))
        if "cont_sent" in df.columns:
            for i, row in df.iterrows():
                #contents.append(row["token"])
                texts.extend(row["cont_sent"])
    
    w2v_model_file = train_w2v(texts,model_folder)
    d2v_model_file = train_d2v(texts, model_folder)

    return w2v_model_file, d2v_model_file

if __name__ == "__main__":
    in_folder = "Processed_data"
    vector_folder = create_folder_path("Vector")
    print("\n----------------------------- Training word2vec and doc2vec models -----------------------\n")
    w2v_model_file, d2v_model_file = train_model(in_folder)
    print(w2v_model_file)
    print(d2v_model_file)
    #w2v_model_file = "Model/w2v_model"
    #d2v_model_file = "Model/d2v_model"

#def glove_to_w2v(glove_input_file, model_folder):
#    """
#    Transform Glove vector format into Word2vec format
#    glove_input_file: file contains GloVe vectors
#    model_folder: Path to saved  model
#    Return: 
#    word2vec_output_file: path to converted Word2vec vectors
#    """
#    word2vec_output_file = model_folder+"/"+glove_input_file+".word2vec"
#    glove2word2vec(glove_input_file, word2vec_output_file)
#    return word2vec_output_file