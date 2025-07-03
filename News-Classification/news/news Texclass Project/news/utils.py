import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup, AdamW, BertModel
from string import punctuation
import re
# # Download NLTK data
# nltk.download('stopwords')
# nltk.download('WordNet')
def data_process():
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test',return_X_y=True)

    label_list = list(newsgroups_train.target_names)
    label_num = len(list(newsgroups_train.target_names))
    print('label_list：',label_list)
    print('class_num：',label_num)
    newsgroups_train = fetch_20newsgroups(subset='train', return_X_y=True)
    train_data, train_label = [],[]
    test_data, test_label = [],[]
    max_len = 0
    for i in range(len(newsgroups_train[0])):
        # TODO：数据预处理
        # 除标点符号，并将文本拆分为单词
        word = re.sub(r'[{}]+'.format(punctuation), ' ', newsgroups_train[0][i].lower().replace("\n", " "))
        word = [w for w in word.split(' ') if w != '']
        # # 添加停用词去除
        # word = [w for w in word if w not in stopwords.words('english')]
        # 添加词形还原
        # lemmatizer = WordNetLemmatizer()
        # word = [lemmatizer.lemmatize(w) for w in word]

        max_len = max(max_len, len(word))
        train_data.append(word)
        train_label.append(newsgroups_train[1][i])

    for i in range(len(newsgroups_test[0])):
        word = re.sub(r'[{}]+'.format(punctuation), ' ', newsgroups_test[0][i].lower().replace("\n", " "))
        word = [w for w in word.split(' ') if w != '']
        max_len = max(max_len, len(word))
        test_data.append(word)
        test_label.append(newsgroups_test[1][i])

    print('max_length',max_len)
    max_len = min(max_len, 256)
    print('max_length', max_len)

    return train_data, train_label,test_data, test_label, label_list, label_num, max_len

def data_loader(data, label, tokenizer, max_len):
    data_len = len(data)
    text_input = torch.zeros((data_len, max_len)).long()
    mask_input = torch.zeros((data_len, max_len), dtype=torch.uint8)
    data_label = torch.zeros(data_len).long()

    text_dd = []
    for i in range(data_len):

        data_label[i] = label[i]
        text = tokenizer.convert_tokens_to_ids(['[CLS]'] + data[i][:254] + ['[SEP]'])
        text_input[i][:len(text)] = torch.tensor(text)
        mask_input[i][:len(text)] = 1
    print('text_input.size():',text_input.size(),'text_input.size():', mask_input.size(), 'data_label.size()',data_label.size())
    return TensorDataset(text_input, mask_input, data_label)

if __name__=='__main__':
    train_data, train_label,test_data, test_label, label_list, label_num, max_len = data_process()

    tokenizer = AutoTokenizer.from_pretrained(r'E:\trasformer\bert-base-uncased', use_fast=True)
    config = AutoConfig.from_pretrained(r'E:\trasformer\bert-base-uncased')
    print(train_data[0])
    print(train_label[0])
    print('len(train_data)',len(train_data))
    print('len(test_data)',len(test_data))
    train_tensor = data_loader(train_data, train_label, tokenizer, max_len)
    test_tensor = data_loader(test_data, test_label, tokenizer, max_len)