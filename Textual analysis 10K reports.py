#=============================================================================#
#======================= Task 3. Textual Analysis  ===========================#
#=============================================================================#
#Download 10-k reports for all listed companies in 2023 and use uncertainty 
#wordlist from Loughran and McDonald (2010) 
#to measure each annual report’s uncertainty tone. 
#Then conduct an event-based regression to test whether higher uncertainty 
#will lead to higher daily stock return. 
#=============================================================================#

 -*- coding: utf-8 -*-
"""

@author: chend
"""

# -*- coding: utf-8 -*-
"""
Keyword Count
@author: chend


LoughranMcDonald_SentimentWordlists:  https://sraf.nd.edu/loughranmcdonald-master-dictionary/
"""

import os
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize     # https://www.tutorialspoint.com/python/python_word_tokenization.htm
from nltk.corpus import stopwords           # https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
from nltk.stem import PorterStemmer         # https://www.geeksforgeeks.org/python-stemming-words-with-nltk/
from tqdm import tqdm


################################################################################################
#########################################################################
'''Function Definition'''

def get_file_list(path):
    file_list = []
    for root, dirs, files in os.walk(path): # files contain: (1) all sub-directory; and (2) all files within the directory
        for file in files:
            if file.lower().endswith('txt'): #identify if the file is txt form
                file_list.append(os.path.join(root,file)) #add to the file list
    return file_list



# define function to read lines from a txt file
def get_non_empty_lines(filePath):
    lines = open(filePath,errors='ignore').read().splitlines()
    non_empty_lines = list()        # set up a non_empty_lines list to store all lines
    for line in lines:
        if line.strip():  # remove whitespace character: https://en.wikipedia.org/wiki/Whitespace_character
            # the if condition means: if the line still has characters after removing whitespace characters
            line = line.upper()     # make letter into upper case: https://www.geeksforgeeks.org/isupper-islower-lower-upper-python-applications/
            non_empty_lines.append(line)    # append the line to the non_empty_lines list as an element
    # print(non_empty_lines)
    return non_empty_lines

# define function to get cik (CENTRAL INDEX KEY)
def get_cik(non_empty_lines):
    cik = ''
    for line in non_empty_lines:
        match = re.search('CENTRAL INDEX KEY:', line)
        if match:
            cik = re.sub('[^0-9]', '', line)    # remove non-number characters from the line, because we only need cik
            # print(cik)
    return cik


# define function to get financial period end date
def get_date(non_empty_lines):
    datadate = ''
    for line in non_empty_lines:
        match = re.search('FILED AS OF DATE:', line) #Another date: COMFORMED PERIOD OF REPORT
        if match:
            datadate = re.sub('[^0-9]', '', line)    # remove non-number characters from the line
            # print(datadate)
    return datadate


# define function to read txt as string
def get_string(filePath):
    content_string = open(filePath,errors='ignore').read().replace('\n', ' ')    # read the 10-K as a string
    # https://stackoverflow.com/questions/8369219/how-to-read-a-text-file-into-a-string-variable-and-strip-newlines
    # print(content_string)
    return content_string


# define function to remove stop words from the content
def clean_text(content_string):

    # initial prep
    #remove content between angle brackets
    content_string = re.sub(r'\<.*?\>', ' ', content_string)    # remove tags (https://tutorialedge.net/python/removing-html-from-string/)
    # print(content_string)

    content_string = re.sub('[0-9]', ' ', content_string)  # remove number characters from the line
    # print(content_string)
    content_string = re.sub(r'[^\w\s]', ' ', content_string) # replace non(^) word characters or whitespaces with an empty string
    #[^...] caret means except
    # \w stands for any alphanumeric character, i.e., a-z,A-Z, 0-9
    # \s stands for space, tab or other whitespace characters

    content_string = content_string.lower()     # lower-case all word
    # print(content_string)

    # step1: word tokenization:
    # https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
    # https://pythonprogramming.net/stop-words-nltk-tutorial/
    word_tokens = word_tokenize(content_string)
    # word_tokens = sent_tokenize(content_string)
    # print(word_tokens)

    #step2: remove stop words:
    #A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore,
    # both when indexing entries for searching and when retrieving them as the result of a search query.
    #nltk_stopwords = list(stopwords.words('english'))
    #print(nltk_stopwords)   # first let's see what are the stop words defined by NLTK

    filter = []
    for word in word_tokens:
        if word not in nltk_stopwords:
            filter.append(word)
    #print(filter)

    # compare how many stop words are removed:
    # print(len(nltk_stopwords))
    # print(len(word_tokens))
    # print(len(filter))

    #step3: stemming: the process of reducing each word to its root or base
    #Stemming is the process of reducing inflection in words to their root forms such as mapping a group of words to the same stem even if the stem itself is not a valid word in the Language
    #e.g. “fishing,” “fished,” “fisher” all reduce to the stem “fish”

    porter = PorterStemmer()
    porterstem_filter = [porter.stem(word) for word in filter]
    #print(porterstem_filter)
    
    #return filter
    return porterstem_filter

    #return ' '.join(porterstem_filter)porterstem_filter
    # return ' '.join(filter)


'''This is the main function to count keywords'''
def find_keyword(file_path):
    #global keywords
    keywords = dfstem_Neg.set_index('Negwords')['count'].to_dict()
    #keywords = keywords.fromkeys(stem_Negwords, 0)  #Initial the dict
    
    file_name = os.path.basename(file_path) #load the file
    
    content_string = get_string(file_path)
    
    porterstem_filter = clean_text(content_string)
    
    total_words = len(porterstem_filter) #denominator
    
    content_after_stem = ' '.join(porterstem_filter)

    #Refer to:https://towardsdatascience.com/tf-idf-explained-and-python-sklearn-implementation-b020c5e83275
    
    for word in porterstem_filter:
        if word.upper() in keywords:
            keywords[word.upper()]+=1
    return file_name, keywords, total_words, content_after_stem


################################################################################
##################################################################################################################

LM_master = pd.read_csv('Loughran-McDonald_MasterDictionary_1993-2021.csv')

FinNeg = LM_master[['Word']][LM_master['Negative']!=0]
FinNeg.columns = ['Negative_words']

nltk_stopwords = open('stopwords.txt','r').read()
nltk_stopwords = nltk_stopwords.split('\n')

#Set the column index
temp = FinNeg['Negative_words'].tolist()
porter = PorterStemmer()
stem_Negwords = [porter.stem(word) for word in temp]
stem_Negwords = list(set(stem_Negwords))   #Delete the duplicates

dfstem_Neg = pd.DataFrame(stem_Negwords)
zero_list = [0]*len(stem_Negwords)
dfstem_Neg['1'] = zero_list
dfstem_Neg.columns = ['Negwords','count']
dfstem_Neg['Negwords']=dfstem_Neg['Negwords'].str.upper()


#Initial list
data = []
cik_list = []
date_list = []
file_index = []
cum = []
keywords_count = []
percentage = []
corpus = []

for path in tqdm(get_file_list(os.getcwd()), desc="Processing Files"): #get current work dir.
    file_name, word_distribution, total_words, content_after_stem = find_keyword(path) #Find word distribution for each files in the dir.
    #print ('%s, %s' % (file_name, word_contribution))
    non_empty_lines = get_non_empty_lines(path)
    
    cik = get_cik(non_empty_lines)
    
    date = get_date(non_empty_lines)
    
    cik_list.append(cik)
    
    date_list.append(date)
    
    cum.append(total_words)
    
    corpus.append(content_after_stem)
    
    for k in word_distribution:
        data.append(word_distribution[k])


output = np.array(data).reshape(int(len(data)/len(stem_Negwords)),len(stem_Negwords))  #Construct a table to save all the keyword distributions of all 10-Ks
                                                           #Here we have 916 negative words, then the denominator is equal to 916
keywords_count = np.sum(output,axis=1)    #Compute the sum of all negative words of a single document
percentage = [a/b for a, b in zip(keywords_count,cum)]   #percentage = keywords_count/total_words
percentage = [i*100 for i in percentage]  #transfer into percentage
output = np.c_[cik_list,date_list,keywords_count, cum, percentage,output] #merge the data

column = ['cik','date','negwords','cum','proportion']   #CIK: central index key, date: filing date, percentage: sum of negative words of a single file
ind = column+stem_Negwords  #construct the column index
pd_output = pd.DataFrame(output, columns=ind)
final_output = pd_output.iloc[2:,:].reset_index(drop = True) #Remove the first two columns which are not the 10-k reports.
final_output = final_output[['cik','date','negwords','cum','proportion']]
final_output.to_csv('final_output.csv')

# FINISHED, we use STATA to run the regression

# STATA code: sqreg excess_ret_t1 proportion log_size, quantile(0.2 0.4 0.6 0.8 1)
