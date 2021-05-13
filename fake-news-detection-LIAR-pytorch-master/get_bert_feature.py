from transformers import BertTokenizer, BertModel
import pandas as pd
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk import tokenize
import pickle
from tqdm import tqdm
import os

def return_cls(model,tokenizer,text_list):
    # text = "Replace me by any text you'd like."
    cls_list=[]
    
    for text in text_list:
        text = "[CLS] " + text + " [SEP]"
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        cls_list.append(output.pooler_output) # (1,768)
    
    return cls_list

def get_top_wiki_sentences(speaker,statement,topK=3):
    spk=speaker # "scott-surovell"
    try:
        spk_wiki_name = wikipedia.search(spk)
    except:
        return None
    if len(spk_wiki_name)==0:
        return None
    else:
        spk_wiki_name = spk_wiki_name[0]
    # wiki_page = wikipedia.page(spk_wiki_name,auto_suggest=False)
    try:
        wiki_page = wikipedia.page(spk_wiki_name,auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        # corpus = []
        # print(f'ambiguous query [{spk_wiki_name}], select the first three pages: {e.options[:3]}')
        # for i in range(3):
        #     s = e.options[i] # pick the first
        #     wiki_page = wikipedia.page(s)
        #     corpus += tokenize.sent_tokenize(wiki_page.content)
        # random_page = wikipedia.page(e.options)
        # print(f'ambiguous query [{spk_wiki_name}], select a random unambiguos page: {random_page.title}')
        # corpus = tokenize.sent_tokenize(random_page.content)
        return None
    else:
        corpus = tokenize.sent_tokenize(wiki_page.content)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    test_X = vectorizer.transform([statement])
    
    sim_scores = np.dot(test_X.toarray(),X.toarray().T)[0] # the higher the closer
    idxs = np.argsort(sim_scores)[::-1]
    top_sents = np.array(corpus)[idxs[:topK]].tolist() # sorted scores
    return top_sents

def get_top_wiki_feature(train_filename):
    train_file = open(train_filename, 'rb')
    lines = train_file.read()
    lines = lines.decode("utf-8")

    table = []
    for line in lines.strip().split('\n'):
        tmp = line.strip().split('\t')
        while len(tmp) < 16:
            tmp.append('')
        table.append([tmp[i].strip() for i in range(16)])
    df_table = pd.DataFrame(table)
    df_table.columns = ["id","json_ID","label","statement","subjects","speaker","speaker_title","state","party",
                    "barely_true_counts","false_counts","half_true_counts","mostly_true_counts","pants_on_fire_counts",
                    "context","justification"]
    
    # continue processing
    if os.path.isfile(train_filename+".top_wiki_top_sents"):
        with open(train_filename+".top_wiki_top_sents",'rb') as f:
            data_dicts = pickle.load(f)
    else:
        data_dicts={}

    ambiguous_spk_statement_count = 0
    for index, row in tqdm(df_table.iterrows()):
        # print(row['json_ID'], row['speaker'])
        json_id = row['json_ID']
        if json_id in data_dicts:
            continue
        statement = data_dicts[json_id]['statement']
        speaker = data_dicts[json_id]['speaker']
        data_dicts[json_id]={
            'speaker':speaker,
            'statement':statement,
        }
        top_sents = get_top_wiki_sentences(speaker,statement, topK=3)
        if top_sents is None:
            ambiguous_spk_statement_count+=1
            top_sents = [statement,statement,statement]
            print(f'statement id {json_id}, dont have wikipage or ambiguous query [{speaker}],skip and use duplicate statement sentences')
        data_dicts[json_id]['top_wiki_sents'] = top_sents

        with open(train_filename+".top_wiki_top_sents",'wb') as f:
            pickle.dump(data_dicts,f)

    print(f"total {ambiguous_spk_statement_count} statements get meaningless wiki top sentences")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    count = 0
    for json_id in tqdm(data_dicts):
        # top_sents = get_top_wiki_sentences(data_dicts[json_id]['speaker'],data_dicts[json_id]['statement'],topK=3)
        # data_dicts[json_id]['top_wiki_sents'] = top_sents
        data_dicts[json_id]['top_wiki_bert_features'] = return_cls(model,tokenizer,data_dicts[json_id]['top_wiki_sents'])
        if count==0:
            print(f"top sent:\n{data_dicts[json_id]['top_wiki_sents']}")
            print(f"top sent features:\n{data_dicts[json_id]['top_wiki_bert_features']}")
        count+=1

        with open(train_filename+".top_wiki_top_feats",'wb') as f:
            pickle.dump(data_dicts,f)

    print(f"get total {count}*3 features")


def main():
    # related_file=None
    get_top_wiki_feature(train_filename = 'train2.tsv')
    get_top_wiki_feature(train_filename = 'val2.tsv')
    get_top_wiki_feature(train_filename = 'test2.tsv')


if __name__ == "__main__":
    main()