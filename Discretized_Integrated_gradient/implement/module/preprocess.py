import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import json
import os

from konlpy.tag import Okt
from nltk import FreqDist

df=pd.read_csv('./.data/corpus/corpus.csv',encoding='cp949')
df = df[df['news/sentence']==0]

df['기업'].unique()

df = df[df['기업'].isin([1, 2, 4, 5])]
corpus=df['content_new'].to_list()
sentiment_ = df['기업'].to_list()

sentiment=[]

for s in sentiment_:
    if s in [1, 4]:
        sentiment.append(1)
    else:
        sentiment.append(0)

p = np.array(sentiment).mean()
plt.bar(['positive', 'negative'], [p, 1-p],
        width=0.4)

def clean_korean(sent):
    if type(sent) == str:
        h = re.compile('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]+')
        result = h.sub('', sent)
        result=result.replace('\n','').strip()
    else:
        result = ''
    return result

corpus_clean = [clean_korean(i) for i in corpus]

okt = Okt()

preprocess_corpus = []

for x, y in tqdm(zip(corpus_clean, sentiment)):
    sent = okt.pos(x)
    # sent_ = []
    # for s in sent:
    #     if s[1] in use_tag:
    #         if s[0] in up_word:
    #             sent_.append(s[0] + '_{}'.format(y))
    #         elif s[0] in down_word:
    #             sent_.append(s[0] + '_{}'.format(y))
    #         elif len(s[0]) > 1:
    #             if s[1] == 'Adjective':
    #                 sent_.append(s[0] + '_{}'.format(y))
    #             else: 
    #                 sent_.append(s[0])
    # preprocess_corpus.append(sent_)
    # sent = [s[0] + '_{}'.format(y) for s in sent if s[1] in use_tag and len(s[0]) > 1]
    # preprocess_corpus.append(sent)
    # sent = [s[0] for s in sent if s[1] in use_tag and len(s[0]) > 1]
    sent = [s[0] for s in sent if s[1] != 'Josa']
    preprocess_corpus.append(sent)

vocab = FreqDist(np.hstack(preprocess_corpus))

vocab.values()

totalWords = sum([freq**(3/4) for freq in vocab.values()])

word2prob = {word : freq**(3/4) / totalWords for word, freq in vocab.items()}

word2freq = {word : freq for word, freq in vocab.items()}

word2idx = {word : index + 1 for index, word in enumerate(vocab)}
word2idx['pad'] = 0

idx2word = {i:x for x,i in word2idx.items()}

vocab_size=len(word2idx)
os.getcwd()

with open("./assets/word2idx.json", "w") as f:
    json.dump(word2idx, f)
with open("./assets/word2prob.json", "w") as f:
    json.dump(word2prob, f)
with open("./assets/word2freq.json", "w") as f:
    json.dump(word2freq, f)
with open("./assets/idx2word.json", "w") as f:
    json.dump(idx2word, f)

freq_threshold = 10
sequences = []
for sent in tqdm(preprocess_corpus):
    sequences.append([word2idx.get(x) for x in sent if word2freq.get(x) > freq_threshold])


max_len=50

'''CBOW context, target 생성'''
#targets = [] 
#contexts = []
all_contexts = []
freqs = []
labels = []

for num in tqdm(range(len(sequences))):
    sent = sequences[num]
    l = len(sent) # 주어진 문장의 길이
    '''
    for index in range(l):
        
        s = index - window_size # window 시작 위치
        e = index + window_size + 1 # window 끝 위치
        
        # context
        context = []
        for i in range(s, e): 
            if 0 <= i < l and i != index: # window가 주어진 문장의 길이를 벗어나지 않고, 중심에 있는 단어(target)가 아닐 경우
                context.append(sent[i])
        if len(context) < window_size * 2: # padding
            context = context + [0] * (window_size * 2 - len(context))
        contexts.append(context)
        
        # positive pair
        targets.append(sent[index])
        
        # negative pair
        # negative.append(word2idx.get(np.random.choice(list(word2prob.keys()), p=list(word2prob.values()))))
        '''
        
    if l < max_len: # padding
        sent_ = sent + [0] * (max_len - l)
        freqs.append([word2freq.get(idx2word.get(x), 0) for x in sent_])
        all_contexts.append(sent_)
    else:
        sent_ = sent[:max_len]
        freqs.append([word2freq.get(idx2word.get(x), 0) for x in sent_])
        all_contexts.append(sent_)
    
    labels.append(float(sentiment[num]))

use_data = np.array(all_contexts)
freqs = np.array(freqs)
labels = np.array(labels)

np.save('./.data/sentence.npy',use_data)
np.save('./.data/labels.npy',labels)
np.save('./.data/freqs.npy',freqs)

