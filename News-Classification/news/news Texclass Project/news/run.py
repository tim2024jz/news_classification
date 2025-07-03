import sys

import torch.nn.functional as F
import torch
import logging
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup, AdamW, BertModel
from sklearn.metrics import classification_report
import numpy as np
from utils import data_process, data_loader
import matplotlib.pyplot as plt


class Word_BERT(nn.Module):
    def __init__(self, label_num=20):
        super(Word_BERT, self).__init__()
        self.bert = BertModel.from_pretrained(r'E:\trasformer\bert-base-uncased')
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, label_num)
        )

    def forward(self, word_input, masks):
        output = self.bert(word_input, attention_mask=masks)
        # sequence_output = output.last_hidden_state
        # print(sequence_output.size())
        pool = output.pooler_output
        # print(pool.size())
        out = self.out(pool)

        return out


def train(train_data, test_data, label_num, epochs=90):

    train_batch_size =64
    test_batch_size =64
    train_iter = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    test_iter = DataLoader(test_data, shuffle=False, batch_size=test_batch_size)

    model = Word_BERT(label_num)
    model.to(torch.device('cuda:0'))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    no_bert = ['bert']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    ((not any(nd in n for nd in no_decay)) and any(nd in n for nd in no_bert))], 'weight_decay': 0.01,
         'lr': 8e-5},
        {'params': [p for n, p in param_optimizer if
                    ((any(nd in n for nd in no_decay)) and any(nd in n for nd in no_bert))], 'weight_decay': 0.0,
         'lr': 8e-5},
        {'params': [p for n, p in param_optimizer if
                    ((not any(nd in n for nd in no_decay)) and (not any(nd in n for nd in no_bert)))],
         'weight_decay': 0.01, 'lr': 1e-2},
        {'params': [p for n, p in param_optimizer if
                    ((any(nd in n for nd in no_decay)) and (not any(nd in n for nd in no_bert)))], 'weight_decay': 0.0,
         'lr': 1e-2}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, correct_bias=True)
    warm_ratio = 0.05
    print("train_batch_size", train_batch_size)
    print("gradient_steps", 1)
    # print(len(train_data))
    total_steps = (len(train_data) // train_batch_size + 1) * epochs

    print("total_steps", total_steps)
    print('--------------------------------')
    print('-------start train model--------')
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_ratio * total_steps,
                                                num_training_steps=total_steps)

    model.train()
    max_f1 = 0
    # Lists for storing metrics
    train_losses = []
    test_f1_scores = []
    test_precisions = []
    test_recalls = []
    for epoch in range(epochs):
        loss_n = []
        for step, batch in enumerate(train_iter):
            text_input, mask_input, label_t = batch
            text_input, mask_input, label_t = text_input.cuda(), mask_input.cuda(), label_t.cuda()
            out = model(text_input, mask_input)

            loss = F.cross_entropy(out.view(-1, label_num), label_t.view(-1))
            loss_n.append(loss.item())

            loss.backward()

            if (step + 1) % 32== 0 or (step + 1) == len(train_iter):
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        # Calculate and store training loss
        epoch_train_loss = np.sum(loss_n) / len(loss_n)
        train_losses.append(epoch_train_loss)
        print("epoch", epoch + 1, "--loss：", epoch_train_loss)

        if (epoch + 1) >= 6 and (epoch + 1) % 6 == 0:
            model.eval()
            with torch.no_grad():
                out_l = [[], []]
                for batch in test_iter:
                    text_input, mask_input, label_t = batch
                    text_input, mask_input, label_t = text_input.cuda(), mask_input.cuda(), label_t.cuda()
                    out = model(text_input, mask_input)
                    out = out.argmax(dim=-1).cpu().numpy().tolist()

                    labels = label_t.cpu().numpy().tolist()
                    out_l[0] += out
                    out_l[1] += labels
            result = classification_report(out_l[1], out_l[0], zero_division=0)
            print(result)

            result = classification_report(out_l[1], out_l[0], zero_division=0, output_dict=True)
            if result["weighted avg"]['f1-score'] > max_f1:
                max_f1 = result["weighted avg"]['f1-score']
                # torch.save(model.state_dict(), "./model.pth")
            model.train()

    print("max_f1", max_f1)
    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    # Plot test F1 score
    plt.subplot(1, 3, 2)
    if test_f1_scores:
        plt.plot(range(5, epochs + 1, 5), test_f1_scores, label='Test F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('Test F1 Score Over Epochs')
        plt.legend()

    # Plot test precision and recall
    plt.subplot(1, 3, 3)
    if test_precisions and test_recalls:
        plt.plot(range(5, epochs + 1, 5), test_precisions, label='Test Precision')
        plt.plot(range(5, epochs + 1, 5), test_recalls, label='Test Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title('Test Precision and Recall Over Epochs')
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    train_data, train_label, test_data, test_label, label_list, label_num, max_len = data_process()

    tokenizer = AutoTokenizer.from_pretrained(r'E:\trasformer\bert-base-uncased', use_fast=True)
    config = AutoConfig.from_pretrained(r'E:\trasformer\bert-base-uncased')

    print(train_data[0])
    print(train_label[0])
    print('train_len', len(train_data))
    print('test_len', len(test_data))

    train_tensor = data_loader(train_data, train_label, tokenizer, max_len)
    test_tensor = data_loader(test_data, test_label, tokenizer, max_len)

    train(train_tensor, test_tensor, label_num, epochs=90)