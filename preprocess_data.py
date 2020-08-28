import pandas as pd
import nltk
from nltk.util import ngrams
from gensim import corpora, models, similarities
import numpy as np
from tweet_process import Tweet_process
from text_process import Text_process
from tqdm import tqdm
tqdm.pandas()
import os
import glob
from helpers import create_folder_path, write_csv_file, save_pk_file, load_pk_file

def similarity_calculator(headline, content):
    """
    https://dev.to/coderasha/compare-documents-similarity-using-python-nlp-4odp
    https://medium.com/better-programming/introduction-to-gensim-calculating-text-similarity-9e8b55de342d
    """
    dictionary = corpora.Dictionary(content)
    corpus = [dictionary.doc2bow(text) for text in content]
    tf_idf = models.TfidfModel(corpus)
    headline_vector = dictionary.doc2bow(headline)
    headline_vector_tf_idf = tf_idf[headline_vector]
    index = similarities.SparseMatrixSimilarity(tf_idf[corpus],num_features = len(dictionary))
    sim_scores = index[headline_vector_tf_idf]
    #sum_of_sims =(np.sum(sims, dtype=np.float32))
    #avr_sims = float(sum_of_sims/len(content))
    #percentage_of_similarity = round(float((sum_of_sims / len(content)) * 100))

    return sim_scores

def preprocess_text(column):
    """Preprocessing text data 
    column: dataframs column that contains text data
    Return a dataframe containing processed data
    """
    processed_data = pd.DataFrame()
    docs = column.progress_map(lambda text: Text_process(text))
    processed_data["token"] = docs.progress_map(lambda doc:Text_process.tokenizer(doc))
    processed_data["num_token"] = processed_data["token"].progress_map(lambda x: len(x))
    processed_data["avr_token_len"]  = processed_data["token"].progress_map(lambda x: round(sum(len(i) for i in x)/len(x)))

    processed_data["punct"] = docs.progress_map(lambda doc:Text_process.get_punct(doc))
    contr = docs.progress_map(lambda doc:Text_process.get_contr(doc))
    processed_data["num_contr"] = contr.progress_map(lambda x: len(x))
    
    processed_data["pos"] = docs.progress_map(lambda doc:Text_process.pos_tagger(doc))
    processed_data["tag"] = docs.progress_map(lambda doc: Text_process.tag_tagger(doc))
    processed_data["pos_trigram"] = processed_data["pos"].progress_map(lambda token: list(ngrams(token,3)))
    processed_data["pos_fourgram"] = processed_data["pos"].progress_map(lambda token: list(ngrams(token,4)))

    processed_data["max_dep_path"] = docs.progress_map(lambda doc: Text_process.max_dep_path(doc))
    processed_data["dep"] = docs.progress_map(lambda doc:Text_process.dep_parser(doc))
    
    processed_data["arg"] = docs.progress_map(lambda doc: Text_process.get_arg(doc))
    processed_data["root"] = docs.progress_map(lambda doc: Text_process.get_root(doc))
    processed_data["det"] = docs.progress_map(lambda doc: Text_process.get_det(doc))
    processed_data["advmod"] = docs.progress_map(lambda doc: Text_process.get_advmod(doc))
    
    processed_data["verb"] = docs.progress_map(lambda doc: Text_process.get_verb(doc))
    processed_data["nn"] = docs.progress_map(lambda doc: Text_process.get_nn(doc))
    processed_data["adj"] = docs.progress_map(lambda doc: Text_process.get_adj(doc))
    processed_data["pron"] = docs.progress_map(lambda doc: Text_process.get_pron(doc))
    processed_data["adv"] = docs.progress_map(lambda doc: Text_process.get_adv(doc))
    
    processed_data["ent"] = docs.progress_map(lambda doc:Text_process.get_ent(doc))
    processed_data["ent_label"] = docs.progress_map(lambda doc:Text_process.get_ent_label(doc))

    processed_data["chunk_dep"] = docs.progress_map(lambda doc:Text_process.get_chunk_dep(doc))

    processed_data["senti_score"] = docs.progress_map(lambda doc:Text_process.senti_score(doc))

    processed_data["use_question"] = docs.progress_map(lambda doc:Text_process.check_question_form(doc))
    processed_data["use_passive"] = docs.progress_map(lambda doc: Text_process.check_passive(doc))
    processed_data["use_supper"] = docs.progress_map(lambda doc: Text_process.check_supper(doc) )
    processed_data["use_if"] = docs.progress_map(lambda doc: Text_process.check_conditional(doc))
    processed_data["use_list"] = docs.progress_map(lambda doc: Text_process.check_listicle(doc))
    processed_data["use_modal"] = docs.progress_map(lambda doc: Text_process.check_modal(doc))
    
    #processed_data["dep_bigram"] = processed_data["dep"].progress_map(lambda token: list(ngrams(token,2)))
    #processed_data["dep_trigram"] = processed_data["dep"].progress_map(lambda token: list(ngrams(token,3)))
    #use_det = docs.progress_map(lambda doc: Text_process.check_det(doc))
    #lemma = docs.progress_map(lambda doc:Text_process.lemmatizer(doc))
    #lemma_bigram = token.progress_map(lambda token: list(ngrams(lemma, 2)))
    #lemma_trigram = token.progress_map(lambda token: list(ngrams(lemma, 3)))
    #tokenized_sent =  docs.progress_map(lambda doc:Text_processing.tokenised_sentencier(doc))
    #nn_chunk = docs.progress_map(lambda doc:Text_process.get_nn_chunk(doc))
    #ent = docs.progress_map(lambda doc:Text_process.get_ent(doc))
    #pos_bigram = pos.progress_map(lambda token: list(ngrams(token, 2)))
    #dep_sub =  docs.progress_map(lambda doc: Text_process.get_dep_subtree(doc))
    #use_comp = tag.progress_map(lambda x: True if "JJC" in x or "RBR" in x else False)

    return processed_data

def preprocess(in_folder, out_folder):
    """
    Preprocess the data file in in_folder and return the processed data to out_folder
    in_folder: path to folder containing the data file
    out_folder: path to folder in which the processed data file is saved
    
    """
    processed_folder = create_folder_path(out_folder)

    for file_name in glob.glob(os.path.join(in_folder, '*.pk')):
        print(f"Reading {file_name}")
        load_data = load_pk_file(file_name)
        headlines = load_data["targetTitle"]
        processed_headlines = preprocess_text(headlines)
        processed_data = pd.concat([headlines, processed_headlines], axis=1)
        processed_data["truthClass"] = load_data["truthClass"]
                
        if len(load_data.columns) > 2:
            processed_data["targetParagraphs"] = load_data["targetParagraphs"]
            docs = processed_data["targetParagraphs"].progress_map(lambda text: [] if text == [] else Text_process(text))
            processed_data["cont_sent"] = docs.progress_map(lambda doc: [] if doc == [] else Text_process.tokenised_sentencier(doc))
            processed_data["cont_num_sent"] = processed_data["cont_sent"].progress_map(lambda x: len(x))
            processed_data["cont_avr_sent_len"] = processed_data["cont_sent"].progress_map(lambda x: 0 if len(x) == 0 else round(sum(len(i) for i in x)/len(x)))
            processed_data["cont_token"] = docs.progress_map(lambda doc: [] if doc == [] else Text_process.tokenizer(doc))
            processed_data ["cont_num_token"] = processed_data["cont_token"].progress_map(lambda x: len(x))
            processed_data["cont_avr_token_len"]  = processed_data["cont_token"].progress_map(lambda x: 0 if len(x) == 0 else round(sum(len(i) for i in x )/len(x)))
    
            processed_data["cont_arg"] = docs.progress_map(lambda doc: Text_process.get_arg(doc))
            processed_data["cont_root"] = docs.progress_map(lambda doc: Text_process.get_root(doc))
          
            processed_data["cont_ent"] = docs.progress_map(lambda doc:Text_process.get_ent(doc))
            processed_data["cont_ent_label"] = docs.progress_map(lambda doc:Text_process.get_ent_label(doc))

            processed_data["cont_senti_score"] = docs.progress_map(lambda doc:Text_process.senti_score(doc))

            sim_scores = []
            for i, row in processed_data[["token","cont_sent"]].iterrows():
                if row["cont_sent"] == []:
                    score = "NA"
                    sim_scores.append(score)
                else:
                    sim_score = similarity_calculator(row["token"], row["cont_sent"])
                    sim_scores.append(sim_score)
                #processed_data.at[i,'sim_score'] = sim_score

            processed_data["sim_score"] = sim_scores
            processed_data["avr_sim_score"] = processed_data["sim_score"].progress_map(lambda score: "NA" if score == "NA" else float(np.sum(score)/len(score)))  
            processed_data["sim_pct"] = processed_data["sim_score"].progress_map(lambda score: "NA" if score == "NA" else round(np.count_nonzero(score)/len(score)*100))

            processed_data.drop(columns=['sim_score'])

        save_file_name = processed_folder+'/'+ os.path.basename(file_name).replace(".pk","")
        #write_csv_file(processed_data,save_file_name)
        save_pk_file(processed_data,save_file_name)
        print(save_file_name)

if __name__ == "__main__":
    preprocess("Data", "Processed_data")
