import os
import torch
import logging
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class BiobertEmbedding(object):
    """
    Encoding from BioBERT model (BERT finetuned on PubMed articles).
    Parameters
    ----------
    model : str, default Biobert.
            pre-trained BERT model
    """

    # def __init__(self, model_path="/home/tjvsonsbeek/Documents/multimodal-VAE/Models/cased_L-12_H-768_A-12/", padding = 256):
    def __init__(self, model_path="/media/tjvsonsbeek/Data1/clinicalBERT/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000", padding = 256):
        if model_path is not None:
            self.model_path = model_path

        self.tokens = ""
        self.sentence_tokens = ""
        # self.model = BertForMaskedLM.from_pretrained(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained(self.model_path, output_hidden_states=True)

        self.model.to(device)
        self.padding = padding


    def process_text(self, text):
        marked_text = "[CLS] " + text + " [SEP]"
        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = self.tokenizer.tokenize(marked_text)

        if self.padding != None:
            tokenized_text=tokenized_text +['[PAD]' for _ in range(self.padding-len(tokenized_text))]
        if len(tokenized_text)>self.padding:
            tokenized_text = tokenized_text[:self.padding]
        return tokenized_text


    def handle_oov(self, tokenized_text, word_embeddings):
        embeddings = []
        tokens = []
        oov_len = 1
        for token,word_embedding in zip(tokenized_text, word_embeddings):
            if token.startswith('##'):
                token = token[2:]
                tokens[-1] += token
                oov_len += 1
                embeddings[-1] += word_embedding
            else:
                if oov_len > 1:
                    embeddings[-1] /= oov_len
                tokens.append(token)
                embeddings.append(word_embedding)
        return tokens,embeddings


    def eval_fwdprop_biobert(self, tokenized_text):

        # Mark each of the tokens as belonging to sentence "1".
        segments_ids = [1] * len(tokenized_text)
        # Map the token strings to their vocabulary indeces.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        segments_tensors = torch.tensor([segments_ids]).to(device)

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()
        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, hidden1, hidden2 = self.model(tokens_tensor, segments_tensors).to_tuple()

        return hidden2


    def word_vector(self, text):

        tokenized_text = self.process_text(text)

        encoded_layers = self.eval_fwdprop_biobert(tokenized_text)

        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # Swap dimensions 0 and 1.

        token_embeddings = token_embeddings.permute(1,0,2)

        # Stores the token vectors, with shape [22 x 768]
        word_embeddings = torch.zeros((self.padding,768))

        # For each token in the sentence...
        for i, token in enumerate(token_embeddings):
            word_embeddings[i,:] = torch.sum(token[-4:], dim=0)

        return word_embeddings



    def sentence_vector(self,text):

        print("Taking last layer embedding of each word.")
        print("Mean of all words for sentence embedding.")
        tokenized_text = self.process_text(text)
        self.sentence_tokens = tokenized_text
        encoded_layers = self.eval_fwdprop_biobert(tokenized_text)

        # `encoded_layers` has shape [12 x 1 x 22 x 768]
        # `token_vecs` is a tensor with shape [22 x 768]
        token_vecs = encoded_layers[11][0]

        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        print("Shape of Sentence Embeddings = %s",str(len(sentence_embedding)))
        return sentence_embedding
