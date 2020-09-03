#pos feature: create a group of fix length array (30) fill with spacy pos vector
#entity feature: same as pos then sum the two array
#mean similarity score of headlines with each sentence in the text
#demonstrative pronoun: this that these those
#Personal pronoun: I, we, you, he, she, they, me, us, you, him, her, them, mine, yours, his, hers, theirs
#Article: the

import numpy as np
import pandas as pd
from text_process import Text_process
from helpers import create_folder_path,load_pk_file,save_pk_file
from sklearn.feature_extraction import DictVectorizer
from sklearn import utils
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

def fix_len_vec(vec, length = 30):
    """
    Resize a vector. If the length of the original vector is shorter than the desire length, pad the vector with 0 at the end.
    If the length of the original vector is longer than the deire length, slice the vector from the begining to the 30th element.
    vec: the original vector that is need to be resize
    length: the desire length

    Return a vector with the desire length.
    """
    if len(vec) < length:
        new_vec = np.pad(vec,(0,(30-len(vec))), mode='constant')
    else:
        new_vec = vec[:30]
    return new_vec

def extract_sty_feat(df, length = 30):
    """
    Extract stylometric features and tranform them into feature vectors.
    df: dataframe that contain the data
    length: the desire length of vectors

    Return a list of feature vectors
    """
    docs = df["targetTitle"].progress_map(lambda text: Text_process(text))
    #turn some sentence feature into vectors
    pos_vector = docs.progress_map(lambda doc: Text_process.tag_vectoriser(doc))
    #resize vectors of sentence features into the same length
    fixed_vec = pos_vector.progress_map(lambda vec: fix_len_vec(vec, length = length))
    #turn the list of resized vectors into a feature vector
    sent_feat = np.asarray(list(fixed_vec))
    sent_feat.shape

    #extract count features
    df["num_arg"] = df["arg"].progress_map(lambda arg: len(arg))
    df["num_root"] = df["root"].progress_map(lambda root: len(root))
    df["num_det"] = df["det"].progress_map(lambda det: len(det))
    df["num_advmod"] = df["advmod"].progress_map(lambda advmod: len(advmod))
    df["num_verb"] = df["verb"].progress_map(lambda verb: len(verb))
    df["num_nn"] = df["nn"].progress_map(lambda nn:len(nn))
    df["num_adj"] = df["adj"].progress_map(lambda adj: len(adj))
    df["num_pron"] = df["pron"].progress_map(lambda pron: len(pron))
    df["num_adv"] = df["adv"].progress_map(lambda adv: len(adv))

    features = []
    for i, row in df.iterrows():
        feat = dict()
        feat["num_token"] = row["num_token"]
        feat["avr_token_len"] = row["avr_token_len"]
        feat["num_contr"] = row["num_contr"]
        feat["max_dep_path"] = row["max_dep_path"]
        feat["num_arg"] = row["num_arg"]
        feat["num_root"] = row["num_root"]
        feat["num_det"] = row["num_det"]
        feat["num_advmod"] = row["num_advmod"]
        feat["num_verb"] = row["num_verb"]
        feat["num_nn"] = row["num_nn"]
        feat["num_adj"] = row["num_adj"]
        feat["num_pron"] = row["num_pron"]
        feat["num_adv"] = row["num_adv"]
        feat["senti_score"] = row["senti_score"]
        feat["use_question"] = row["use_question"]
        feat["use_list"] = row["use_modal"]
       
        features.append(feat)

    dict_vtrz = DictVectorizer(sparse=False)
    #transform extracted features into vectors
    dict_vect = dict_vtrz.fit_transform(features)
    dict_vect.shape

    #concatenate sentence feature vector and count feature vector into sylometric feautre vector
    X_sty = np.concatenate((sent_feat,dict_vect), axis = 1)
    X_sty.shape

    return X_sty

def load_d2v(d2v_model_file):
    dv_model = Doc2Vec.load(d2v_model_file)
    return dv_model

def create_d2v(dv_model, text):
    """
    Create tweet representation using document embeddings
    dv_model: the path to the saved document embedding model
    tweets: a list of tweets from the data file
        
    Return: the name of the file to which feature vectors are saved
    """
    d2v_vec = dv_model.infer_vector(text)

    return d2v_vec


def load_w2v(w2v_model_file):
    # load the Stanford GloVe model
    wv_model = KeyedVectors.load(w2v_model_file)
    return wv_model


def create_w2v(w2v_model,texts, size = 100):
    """
    Create document representation using word embedding
    This function has three positional arguments and one keyword argument
    model_file: the path to the saved word embedding model
    texts: a list of untokenised documents from the data file
    size: the size of word embeddings 
      
    Return: name of the save file that contain the document vectors
    """
    print('Create document representation using word embedding')
    word_embs = []
    wmb_size = size*2+2
    nlp = spacy.load('en_core_web_sm')
    for text in tqdm(list(texts)):
        text_vec = np.zeros(wmb_size)
        doc = nlp(text.lower())
        for token in doc:
            word = str(token.text)
            head = str(token.head)
            if word in w2v_model.wv.vocab and head in w2v_model.wv.vocab:
                token_vec = w2v_model.wv[word]
                pos_vec =  np.array([token.pos])
                dep_vec =  np.array([token.dep])
                head_vec = w2v_model.wv[head]
                word_vec = np.concatenate((token_vec,pos_vec,dep_vec,head_vec))
                text_vec += word_vec
        
        word_embs.append(text_vec)
    
    X_w2v = np.asarray(word_embs)
    
    return X_w2v

def headline_features(in_folder,w2v_model):
    """
    Extract features from headlines
    in_folder: path to folder containing data file
    w2v_model: word embedding model
    """
    
    sty_vecs = []
    w2v_vecs = []
    labels = []

    for file_name in glob.glob(os.path.join(in_folder, '*.pk')):
        print(f"\nReading {file_name}\n")
        df = load_pk_file(file_name)
        sty_vec = extract_sty_feat(df)
        sty_vecs.append(sty_vec)
        w2v_vec = create_w2v(w2v_model,df["targetTitle"])
        w2v_vecs.append(w2v_vec)
        labels.append(df["truthClass"])
    
    print("Concatenating feature vectors")
    X_sty = np.concatenate(sty_vecs, axis = 0)
    print(f"Stylometric: {X_sty.shape}")
    X_w2v = np.concatenate( w2v_vecs, axis = 0)
    print(f"Word2vec: {X_w2v.shape}")
    X_cmb = np.concatenate((X_w2v,X_sty), axis = 1)
    print(f"Combined: {X_cmb.shape}")
    y = np.asarray(pd.concat(labels))

    print("Splitting data")
    print("Stylometric")
    sty_file = "Vector/sty"
    save_pk_file((X_sty,y), sty_file)
        
    print("Word2vec")
    w2v_file = "Vector/w2v"
    save_pk_file((X_w2v,y),w2v_file)

    print("Combined")
    cmb_file = "Vector/cmb"
    save_pk_file( (X_cmb,y), cmb_file)
    
    print("Done")


def content_features(in_folder,w2v_model,d2v_model):
    """
    Extract features from contents
    in_folder: path to folder containing data file
    w2v_model: word embedding model
    d2v_model: document embedding model
    """

    sty_vecs = []
    w2v_vecs = []
    d2v_vecs = []
    labels = []

    for file_name in glob.glob(os.path.join(in_folder, '*.pk')):
        print(f"\nReading {file_name}\n")
        df = load_pk_file(file_name)
        if "targetParagraphs" in df.columns:
            filtered_df = df[df["targetParagraphs"].apply(lambda x: len(x)>0)]
            sty_vec = extract_sty_feat(filtered_df)
            sty_vecs.append(sty_vec)
            w2v_vec = create_w2v(w2v_model,filtered_df["targetTitle"])
            w2v_vecs.append(w2v_vec)
            labels.append(filtered_df["truthClass"])
            sent_vec = filtered_df["cont_sent"].progress_map(lambda x: np.asarray([create_d2v(d2v_model,i) for i in x]))
            avr_sent_vec = sent_vec.progress_map(lambda x:np.mean(x, axis = 0))
            d2v_vec = np.asarray(list(avr_sent_vec))
            
            features = []
            for i, row in filtered_df.iterrows():
                feat = dict()
                feat["cont_num_token"] = row["cont_num_token"]
                feat["cont_avr_token_len"] = row["cont_avr_token_len"]
                feat["cont_senti_score"] = row["cont_senti_score"]
                feat["avr_sim_score"] = row["avr_sim_score"]
                feat["sim_pct"] = row["sim_pct"]
                features.append(feat)
            
            dict_vtrz = DictVectorizer(sparse=False)
            dict_vect = dict_vtrz.fit_transform(features)
            
            d2v_vecs.append(np.concatenate((d2v_vec,dict_vect), axis = 1))

    print("Concatenating feature vectors")
    X_sty = np.concatenate(sty_vecs, axis = 0)
    print(f"Stylometry: {X_sty.shape}")
    X_w2v = np.concatenate(w2v_vecs, axis = 0)
    print(f"Word2vec: {X_w2v.shape}")
    X_d2v = np.concatenate(d2v_vecs, axis = 0)
    print(f"Doc2vec: {X_d2v.shape}")
    X_cmb = np.concatenate((X_w2v,X_sty,X_d2v), axis = 1)
    print(f"Combined: {X_cmb.shape}")
    y = list(pd.concat(labels))

    print("Splitting data")
    print("Doc2vec")
    cmb_file = "Vector/d2v"
    save_pk_file((X_cmb,y), cmb_file)
    
    print("Done")

if __name__ == "__main__":
    in_folder = "Train"
    vector_folder = create_folder_path("Vector")
    print("\n----------------------------- Loading word2vec and doc2vec models -----------------------\n")
        
    w2v_model_file = "".join([f for f in glob.glob(os.path.join("Model/*_model")) if "w2v" in f])
    d2v_model_file = "".join([f for f in glob.glob(os.path.join("Model/*_model")) if "d2v" in f])
    w2v_model = load_w2v(w2v_model_file)
    d2v_model = load_w2v(d2v_model_file)

    print("\n----------------------------- Creating headline feature vectors -----------------------\n")
    headline_features(in_folder,w2v_model)
    print("\n----------------------------- Creating content feature vectors -----------------------\n")
    content_features(in_folder,w2v_model,d2v_model)