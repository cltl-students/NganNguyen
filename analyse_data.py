import pandas as pd
import os
import glob
from helpers import load_pk_file, save_pk_file, write_csv_file, merge_list_of_lists,create_folder_path
from tqdm import tqdm
tqdm.pandas()
from helpers import load_pk_file, merge_list_of_lists, create_folder_path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def get_most_freq(column, n):
    """
    Get most n frequent items in a column, calculate the frequency contribution of each element
    column: name of the column to be plotted
    """
    #count the frequency of each item in a column
    count = Counter()
    column.apply(lambda x: count.update(Counter(x)))
    #calculate the percentage of occurance over the total number of document
    percentage = [( i, count[i]/len(column)*100.0 ) for i in count]
    df = pd.DataFrame(percentage, columns=['item', 'percentage'])
    most_freq = df.sort_values(by="percentage", ascending=False).head(n)
    
    return most_freq

def cal_pct(val, list_val):
    pct = int(val/100.*np.sum(list_val))
    return "{:.1f}%".format(pct)

class Plot_data:

    def __init__(self):
        pass

    def plot_num_feat(self, column, out_folder, group_by='truthClass'):
        truthClass = list(self.truthClass.unique())
        group = self.groupby([column, group_by]).size().unstack()
        for i in truthClass:
            group["pct_"+i] = group[i].map(lambda x: x/group[i].sum())
            del group[i]
        title = f"Plotting the {column} in clickbait and non-clickbait"
        plot = group.plot(kind = "bar",figsize=[18,10], title = title, colormap ='Paired')
        save_file = f'{out_folder}/plot_{column}.pdf'
        plot.figure.savefig(save_file)
        plt.close()

        return save_file

    def plot_check_feat(self, column,out_folder, group_by = "truthClass"):
        """
        Plotting contribution
        """
        group = self.groupby([column,group_by]).size().unstack()
        fig, axs = plt.subplots(1,2,figsize=(12, 6), subplot_kw=dict(aspect="equal"))
        #fig.suptitle(f"Plotting the occurrence of {column} in clickbait and non-clickbait") 

        labels = group.axes[0].tolist()
        explode = (0, 0.1)
        
        for  i, c in enumerate(list(group.columns)):
            wedges, texts, autotexts = axs[i].pie(group[c],explode=explode, autopct=lambda val: cal_pct(val,group[c]), 
                                                        colors= ['#003f5a','#de6600'], textprops=dict(color="w"))

            axs[i].set_title(c,fontdict=dict(fontsize=14,fontweight='bold'), y = -0.1)
            plt.setp(autotexts, size=14, weight="bold")

        axs[1].legend(wedges, labels,loc='upper right')
        
        save_file = f'{out_folder}/plot_{column}.pdf'
        fig.savefig(save_file)
        plt.close()

        return save_file
    
    def plot_freq_feat(self, column, out_folder, n=20):
        """"
        Plotting frequency of occurrence of a feature.
        df: dataframe
        column: name of the column to be plotted
        folder: name of the folder to which the result figure is saved
        Return a horizontal bar chart
        """
        #calculate frequency of occurrence of a feature with each category
        truthClass = list(self.truthClass.unique())
        freq_list = []
        for label in truthClass:
            most_freq = get_most_freq(self[self["truthClass"] == label][column], n)
            freq_list.append(most_freq)
    
        #merge the results of the calculation to compare. Missing values will be filled with Nah
        freq_df = pd.merge(freq_list[0], freq_list[1], on='item', how='outer',suffixes=('_CB','_NO'))
    
        #Plot the merged result
        fig = plt.figure(figsize=(24,12))
        y = np.arange(len(freq_df.index))
        ax = plt.subplot(111)
        ax.barh(y, freq_df["percentage_CB"], height=0.3, color='#003f5a', align='center')
        ax.barh(y-0.3, freq_df["percentage_NO"],height=0.3, color='#de6600', align='center')
        ax.legend((truthClass))
        plt.yticks(y, freq_df["item"], fontsize=12)
        ax.invert_yaxis()
        ax.set_xlabel('Contribution', fontsize=15)
        plt.title(f"Plotting the frequency of occurrence of {column} in clickbait and non-clickbait", fontsize=15)
        #plt.show()
        save_file = f"{out_folder}/plot_{column}.pdf"
        fig.savefig(save_file,orientation='landscape')
        plt.close()

        return save_file
    
def analyse_data(in_folder, out_folder):
    fig_folder = create_folder_path(out_folder)

    for file_name in glob.glob(os.path.join(in_folder, '*.pk')):
        print(f"Analysing {file_name}")
        df = load_pk_file(file_name)
        save_folder = fig_folder + "/" + os.path.basename(file_name).replace(".pk","")
        create_folder_path(save_folder)

        #plotting the number of tokens of each text
        Plot_data.plot_num_feat(df,"num_token",save_folder)
        #Plotting average token lenght
        Plot_data.plot_num_feat(df,"avr_token_len",save_folder)
        #Plotting the use of punct
        Plot_data.plot_freq_feat(df,"punct",save_folder)
        #plotting number of contraction
        Plot_data.plot_num_feat(df,"num_contr",save_folder)
        #plotting the frequency of each part-of-speech
        Plot_data.plot_freq_feat(df,"pos",save_folder)
        #plotting the frequency of each part-of-speech
        Plot_data.plot_freq_feat(df,"tag",save_folder,40)
        #plotting most frequent pos_ngram
        Plot_data.plot_freq_feat(df,"pos_trigram",save_folder,30)
        Plot_data.plot_freq_feat(df,"pos_fourgram",save_folder,30)
        #plotting the longest dependecy path of each text
        Plot_data.plot_num_feat(df, "max_dep_path",save_folder)
        #plotting the frequency of each dependecy
        #Plot_data.plot_freq_feat(df,"dep",save_folder,40)
        #Plot_data.plot_freq_feat(df,"dep_bigram",save_folder,30)
        #Plot_data.plot_freq_feat(df,"dep_trigram",save_folder,30)
        #plotting most frequent subject
        Plot_data.plot_freq_feat(df,"arg",save_folder)
        #plotting most frequent root
        Plot_data.plot_freq_feat(df,"root",save_folder)
        #plotting most frequent determiner
        Plot_data.plot_freq_feat(df,"det",save_folder)
        #plotting most frequent adverb
        Plot_data.plot_freq_feat(df,"advmod",save_folder)
        #plotting the frequency of verbs
        Plot_data.plot_freq_feat(df,"verb",save_folder)
        #plotting the frequency of noun
        Plot_data.plot_freq_feat(df,"nn",save_folder)
        #plotting the frequency of adj
        Plot_data.plot_freq_feat(df,"adj",save_folder)
         #plotting most frequent pronoun
        Plot_data.plot_freq_feat(df,"pron",save_folder)
        #plotting the frequency of adv
        Plot_data.plot_freq_feat(df,"adv",save_folder)
        #plotting the number of named entities
        Plot_data.plot_freq_feat(df,"ent",save_folder)
        df["num_ent"] = df["ent_label"].progress_map(lambda x: len(x))
        Plot_data.plot_num_feat(df,"num_ent",save_folder)
        #plotting the most frequent types of named entity
        Plot_data.plot_freq_feat(df,"ent_label",save_folder)
        #plotting the frequency of different type of chuck
        Plot_data.plot_freq_feat(df,"chunk_dep",save_folder)
        #plotting sentiment
        Plot_data.plot_num_feat(df,"senti_score",save_folder)
        #plotting texts that in the form of a question
        Plot_data.plot_check_feat(df,"use_question",save_folder)
        #plotting texts that use passive voice
        Plot_data.plot_check_feat(df,"use_passive",save_folder)
        #plotting texts that use supperlative
        Plot_data.plot_check_feat(df,"use_supper",save_folder)
        #plotting the use of if statement
        Plot_data.plot_check_feat(df,"use_if",save_folder)
        #plotting texts that is listicle
        Plot_data.plot_check_feat(df,"use_list",save_folder)
        #plotting modal verbs
        Plot_data.plot_check_feat(df,"use_modal",save_folder)
        
        if "targetParagraphs" in df.columns:
            #plotting the number of tokens of each text
            Plot_data.plot_num_feat(df,"cont_num_token",save_folder)
            #Plotting average token lenght
            Plot_data.plot_num_feat(df,"cont_avr_token_len",save_folder)
            #plotting the number of sentences in content
            Plot_data.plot_num_feat(df,"cont_num_sent",save_folder)
            #Plotting average sentence lenght
            Plot_data.plot_num_feat(df,"cont_avr_sent_len",save_folder)
            #plotting the most frequent named entities
            Plot_data.plot_freq_feat(df,"cont_ent",save_folder)
            #plotting sentiment
            Plot_data.plot_num_feat(df,"cont_senti_score",save_folder)
            #Plotting percentage of sentences in content that is similar to headlines
            df["sim_pct"] = df["sim_pct"].progress_map(lambda x: "NA" if x=="NA" else ("<33%" if x<33 else ("33-66%" if 33<=x<66 else ">66%")))
            Plot_data.plot_num_feat(df,"sim_pct",save_folder)
            #Plotting the average similarity scores between a headline and its content
            df["avr_sim_score"] = df["avr_sim_score"].progress_map(lambda x: "NA" if x == "NA" else ("(0-0.05]" if x<=0.05 else ("(0.05-0.1]" if 0.05<x<=0.1 else ("(0.1-0.2]" if 0.1<x<=0.2 
                                                                    else ("(0.2-0.3]" if 0.2<x<=0.3 else ("(0.3-0.4]" if 0.3<x<=0.4 else ("(0.4-0.5]" if 0.4<x<=0.5 else ">0.5")))))))
            Plot_data.plot_num_feat(df,"avr_sim_score",save_folder)

if __name__ == "__main__":
    analyse_data("Processed_data", "Figures")