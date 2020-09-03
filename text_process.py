import spacy
import pandas as pd
import networkx as nx
import re
from textblob import TextBlob
import numpy as np

class Text_process:
    """Process texts using SpaCy and TextBlob"""
    nlp = spacy.load("en_core_web_sm")
    
    def __init__(self, text):
        self.text = text
        self.doc = self.nlp(self.text)
   
    def tokenizer(self):
        tokens =  [token.text.lower() for token in self.doc]
        return tokens
       
     def sentencier(self):
        sents = [sent.text for sent in self.doc.sents]
        return sents
    
    def tokenised_sentencier(self):
        tokenised_sents = []
        for sent in self.doc.sents:
            tokens = [token.text.lower() for token in sent]
            tokenised_sents.append(tokens)
        return tokenised_sents
    
    def pos_tagger(self):
        pos = [token.pos_ for token in self.doc]
        return pos
    
    def tag_tagger(self):
        tag = [token.tag_ for token in self.doc]
        return tag

    def dep_parser(self):
        dep =  [token.dep_ for token in self.doc]
        return dep

    def get_arg(self):
        arg = [token.lemma_ for token in self.doc if token.dep_ in ["nsubj", "nsubjpass", "obj", "dobj", "iobj"]]
        return arg

    def get_root(self):
        root = [token.lemma_ for token in self.doc if token.dep_ == "ROOT"]
        return root
    
    def get_det(self):
        det = [token.text.lower() for token in self.doc if token.dep_ == "det"]
        return det

    def get_advmod(self):
        advmod = [token.text.lower() for token in self.doc if token.dep_ == "advmod"]
        return advmod

    def get_verb(self):
        verb = [token.lemma_ for token in self.doc if token.tag_.startswith("VB")]
        return verb

    def get_nn(self):
        nn = [token.lemma_ for token in self.doc if token.tag_.startswith("NN")]
        return nn

    def get_adj(self):
        adj = [token.lemma_ for token in self.doc if token.tag_.startswith("JJ")]
        return adj


    def get_pron(self):
        pron = [token.text.lower() for token in self.doc if token.tag_.startswith("PR")]
        return pron
   
    def get_adv(self):
        adv = [token.text.lower() for token in self.doc if token.tag_.startswith("RB")]
        return adv
    
    def get_punct(self):
        punct = [token.text for token in self.doc if token.is_punct]
        return punct
    
    def get_contr(self):
        """
        Find all word contraction in the text
        """
        p = re.compile("[a-zA-Z0-9_]'[a-zA-Z0-9_]+")
        contr = p.findall(self.text)
        return contr

    def get_ent(self):
        ent_text = [ent.text for ent in self.doc.ents]
        return ent_text

    def get_ent_label(self):
        ent_label = [ent.label_ for ent in self.doc.ents]
        return ent_label
        
    def get_nn_chunk(self):
        nn_chunks = [chunk.text for chunk in self.doc.noun_chunks]
        return nn_chunks

    def get_chunk_dep(self):
        chunk_root_dep = [chunk.root.dep_ for chunk in self.doc.noun_chunks]
        return chunk_root_dep

        def senti_score(self):
        senti_score = TextBlob(self.text).sentiment.polarity
        if senti_score > 0.5:
            return "ExPos"
        elif 0 < senti_score <= 0.5:
            return "Pos"
        elif senti_score == 0:
            return "Neu"
        elif -0.5 <= senti_score <0:
            return "Neg"
        else:
            return "ExNeg"
        
    def check_question_form(self):
        fst_token = self.doc[0]
        if fst_token.tag_.startswith("VB") or fst_token.tag_ in ["WP", "MD", "WDT","WRB"] or fst_token.text.lower() == "whoes" :
            return True
        else:
            return False
    
    def check_passive(self):
        """Check if a sentence is in passive or active voice"""
        if True in [token.dep_.endswith("pass") for token in self.doc]:
            return True
        else:
            return False

    def check_listicle(self):
        fst_token = self.doc[0]
        if fst_token.dep_ == "nummod" and fst_token.ent_type_ == "CARDINAL":
            return True
        else:
            return False
    
    def check_conditional(self):
        tokens = [token.text.lower() for token in self.doc]
        if "if" in tokens or "unless" in tokens:
            return True
        else:
            return False

    def check_supper(self):
        tags = [token.tag_ for token in self.doc]
        if "JJS" in tags or "RBS" in tags:
            return True
        else:
            return False

    def check_modal(self):
        tags = [token.tag_ for token in self.doc]
        if "MD" in tags:
            return True
        else:
            return False


    def max_dep_path(self):
        """
    Find all dependency path from the root token(s) and return the length of the longest path
    by building a graph using networkx with each node of the graph is a token in the analysed sentence and the edge connect each node is the dependency
    If there is no connection between two node, the length is assigned as 0
    Reference:
    https://towardsdatascience.com/how-to-find-shortest-dependency-path-with-spacy-and-stanfordnlp-539d45d28239
        """
        # Load spacy's dependency tree into a networkx graph
        edges = []
        roots = []
        for token in self.doc:
            if token.dep_ == "ROOT":
                roots.append(token.text.lower())
            for child in token.children:
                edges.append(('{0}'.format(token.lower_),
                            '{0}'.format(child.lower_)))
        graph = nx.Graph(edges)
    #Get the length and path
        length = []
        for token in self.doc:
            for root in roots:
                try: 
                    length_path = nx.shortest_path_length(graph, source=root, target=token.lower_)
                    length.append(length_path)
        #   print(nx.shortest_path(graph, source=root, target=token.lower_))
                except:
                    length_path = 0
                    length.append(length_path)
        return max(length)

    def get_dep_sub(self):
        edges = []
        for token in self.doc:
            for child in token.children:
                edges.append(('{0}'.format(token.tag_),
                            '{0}'.format(child.tag_)))
        return edges
    
    def tag_vectoriser(self):
        """
        Transform pos tags, dependency tags, named entity tags
        """
        pos_vector = np.asarray([token.tag for token in self.doc],dtype=np.float32)
        dep_vector = np.asarray([token.dep for token in self.doc],dtype=np.float32)
        ent_vector = np.asarray([token.ent_type for token in self.doc],dtype=np.float32)
        
        check_punct = np.asarray([token.is_punct for token in self.doc],dtype=np.float32)
        punct_vfunc = np.vectorize(lambda x: 0 if x== False else 1)
        punct_vector = punct_vfunc(check_punct)

        
        check_is_digit = np.asarray([token.is_digit for token in self.doc],dtype=np.float32)
        is_digit_vfunc = np.vectorize(lambda x: 0 if x== False else 2)
        is_digit_vector = is_digit_vfunc(check_is_digit)
        
        n_left = np.asarray([token.n_lefts for token in self.doc],dtype=np.float32)
        n_right = np.asarray([token.n_rights for token in self.doc],dtype=np.float32)

        feature_vector = pos_vector+dep_vector+ent_vector+n_left+n_right+is_digit_vector+punct_vector

        return feature_vector

#    def check_det(self):
#        tags = [token.tag_ for token in self.doc]
#        if "DT" in tags:
#            return True
#        else:
#            return False
    
#    def check_nn(self):
#        tags = [token.tag_ for token in self.doc]
#        if "NNP" in tags:
#            return True
#        else:
#            return False
    

    #def check_comp(self):
     #   if "JJC" or "RBR" in [token.tag_ for token in self.doc]:
      #      return True
       # else:
        #    return False    

    
    #def check_if(self):
     #   if "if" in [token.text.lower() for token in self.doc]:
      #      return True
       # else:
        #    return False

    #def check_listicle(self):
     #   if self.doc[0].is_digit:
      #      return True
       # else:
        #    return False
    #def check_emoji(self):
    #    emoji_check = self.doc._.has_emoji
    #    return emoji_check
    
    #def get_emoji(self):
    #    emojis = []
    #    for token in self.doc:
    #        if token._.is_emoji:
    #            emojis.append(token._.emoji_desc)
    #    return emojis

    #def lemmatizer(self):
    #    lemma =  [token.lemma_ for token in self.doc]
    #    return lemma

    #def stop_word_filter(self):
        #spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    #    lemma = [token.lemma_ for token in self.doc if not token.is_stop]
    #    return lemma