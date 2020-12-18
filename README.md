# --文本增强
# 前言
此次处理主要使用了nlpaug库中RandomWordAug和TfIdfAug方法。
# [什么是TF-IDF](https://baike.baidu.com/item/tf-idf)
这里介绍主要公式TF−IDF=TF∗IDF<br>
词频 ( TF) 指的是某一个给定的词语在该文件中出现的次数。这个数字通常会被归一化(一般是词频除以文章总词数), 以防止它偏向长的文件。（同一个词语在长文件里可能会比短文件有更高的词频，而不管该词语重要与否。）<br>
逆向文件频率 (inverse document frequency, IDF) IDF的主要思想是：如果包含词条t的文档越少, IDF越大，则说明词条具有很好的类别区分能力。某一特定词语的IDF，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取对数得到<br>
# [nlpaug库](https://github.com/makcedward/nlpaug)
This python library helps you with augmenting nlp for your machine learning projects. Visit this introduction to understand about Data Augmentation in NLP. Augmenter is the basic element of augmentation while Flow is a pipeline to orchestra multi augmenter together.


# RandomWordAug方法参数  此次处理通过对action参数取不同值来达到文本增强目的
action (str) – ‘substitute’, ‘swap’, ‘delete’ or ‘crop’. If value is ‘swap’, adjacent words will be swapped randomly. If value is ‘delete’, word will be removed randomly. If value is ‘crop’, a set of contunous word will be removed randomly.<br>
aug_p (float) – Percentage of word will be augmented.<br>
aug_min (int) – Minimum number of word will be augmented.<br>
aug_max (int) – Maximum number of word will be augmented. If None is passed, number of augmentation is calculated via aup_p. If calculated result from aug_p is smaller than aug_max, <br>will use calculated result from aug_p. Otherwise, using aug_max.<br>
stopwords (list) – List of words which will be skipped from augment operation. Not effective if action is ‘crop’<br>
stopwords_regex (str) – Regular expression for matching words which will be skipped from augment operation. Not effective if action is ‘crop’<br>
target_words (list) – List of word for replacement (used for substitute operation only). Default value is _.<br>
tokenizer (func) – Customize tokenization process<br>
reverse_tokenizer (func) – Customize reverse of tokenization process<br>
name (str) – Name of this augmenter<br>
# TfIdfAug方法参数  
model_path (str) – Downloaded model directory. Either model_path or model is must be provided<br>
action (str) – Either ‘insert or ‘substitute’. If value is ‘insert’, a new word will be injected to random position according to TF-IDF calculation. If value is ‘substitute’, word will<br> be replaced according to TF-IDF calculation<br>
top_k (int) – Controlling lucky draw pool. Top k score token will be used for augmentation. Larger k, more token can be used. Default value is 5. If value is None which means using all<br> possible tokens.<br>
aug_p (float) – Percentage of word will be augmented.<br>
aug_min (int) – Minimum number of word will be augmented.<br>
aug_max (int) – Maximum number of word will be augmented. If None is passed, number of augmentation is calculated via aup_p. If calculated result from aug_p is smaller than aug_max,<br> will use calculated result from aug_p. Otherwise, using aug_max.<br>
stopwords (list) – List of words which will be skipped from augment operation.<br>
stopwords_regex (str) – Regular expression for matching words which will be skipped from augment operation.<br>
tokenizer (func) – Customize tokenization process<br>
reverse_tokenizer (func) – Customize reverse of tokenization process<br>
name (str) – Name of this augmenter<br>

# 导入包
import pandas as  pd
import nlpaug.augmenter.word as naw


# 导入数据
```
train_path = "train_cleaned_v2 - 副本.csv"
test_path= "test_cleaned_v2 - 副本.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
```
# randomword_substitute(text)  同义词替换  
```
def randomword_swap(text):       
    aug1 = naw.RandomWordAug(action='swap') 
    augmented_data1 = aug1.augment(text)    
    return  augmented_data1   
```
# randomword_swap(text) 相邻的单词将被随机交换 
```
def randomword_substitute(text): <br>
    aug2 = naw.RandomWordAug(action='substitute') <br>
    augmented_data2 = aug2.augment(text) <br>
    return  augmented_data2 <br>
```
# randomword_delete(text)  单词将被随机删除
```
def randomword_delete(text):
    aug3 = naw.RandomWordAug(action='delete') 
    augmented_data3 = aug3.augment(text) 
    return  augmented_data3 
```
# 数据处理
```
train['swap']=train['text'].apply(randomword_swap)
train['substitute']=train['text'].apply(randomword_substitute) 
train['delete']=train['text'].apply(randomword_delete) 
test['swap']=test['text'].apply(randomword_swap)
test['substitute']=test['text'].apply(randomword_substitute)
test['delete']=test['text'].apply(randomword_delete) 
```

# TF-IDF model training 以train数据为例，test数据同理，更改路径即可
```
import os
os.environ["MODEL_DIR"] = '../model'
import re
import nlpaug.augmenter.word as naw
import nlpaug.model.word_stats as nmw
import pandas as pd

train_path = "train_cleaned_v2 - 副本.csv"
train = pd.read_csv(test_path)

def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)
```
# Load sample data
```
train_data = test['text']
train_x = train_data.head().values
```
# Tokenize input
```
train_x_tokens = [_tokenizer(x) for x in train_x]
```
# Train TF-IDF model
```
tfidf_model = nmw.TfIdf()
tfidf_model.train(train_x_tokens)
tfidf_model.save('.')
```
# Load TF-IDF augmenter
```
aug = naw.TfIdfAug(model_path='.', tokenizer=_tokenizer)
texts = train_x

def tfidfaug(text):
    aug = naw.TfIdfAug(model_path='.', tokenizer=_tokenizer)
    augmented_data = aug.augment(text)
    return  augmented_data
train['tfidf']=test['text'].apply(tfidfaug)
train.to_csv(train_path)
```
