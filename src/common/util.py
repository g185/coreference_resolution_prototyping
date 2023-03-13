import pandas as pd
import spacy
import tqdm
import hydra
from torch.nn import Module, Linear, LayerNorm, Dropout, ReLU
import torch


def flatten(l):
    return [item for sublist in l for item in sublist]

def to_dataframe(file_path):
    df = pd.read_json(hydra.utils.get_original_cwd() + "/" +file_path , lines=True)

    if 'tokens' in df.columns:
        pass
    elif 'sentences' in df.columns:
        # this is just for ontonotes. please avoid using 'sentences' and use 'text' or 'tokens'
        df['tokens'] = df['sentences'].apply(lambda x: flatten(x))
        df["EOS"] = df["sentences"].apply(lambda x: [len(value) for value in x])
        df["EOS"] = df["EOS"].apply(lambda x: [sum(x[0:(i[0]+1)]) for i in enumerate(x)])
        
    elif 'text' in df.columns:
        nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
        texts = df['text'].tolist()
        df['tokens'] = [[tok.text for tok in doc] for doc in tqdm(nlp.pipe(texts), total=len(texts))]
    else:
        raise NotImplementedError(f'The jsonlines must include tokens/text/sentences attribute')

    if 'speakers' in df.columns:
        df['speakers'] = df['speakers'].apply(lambda x: flatten(x))
    else:
        df['speakers'] = df['tokens'].apply(lambda x: [None] * len(x))

    if 'doc_key' not in df.columns:
        raise NotImplementedError(f'The jsonlines must include doc_key, you can use uuid.uuid4().hex to generate.')

    if 'clusters' in df.columns:
        df = df[['doc_key', 'tokens', 'speakers', 'clusters', 'EOS']]
    else:
        df = df[['doc_key', 'tokens', 'speakers', 'EOS']]

    df = df.dropna()
    df = df.reset_index(drop=True)
    return df



class FullyConnectedLayer(Module):
    def __init__(self, input_dim, output_dim, hidden_size, dropout_prob):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.dense1 = Linear(self.input_dim, hidden_size)
        self.dense = Linear(hidden_size, self.output_dim)
        self.layer_norm = LayerNorm(self.output_dim)
        self.activation_func = ReLU()
        self.dropout = Dropout(self.dropout_prob)

    def forward(self, inputs):
        temp = inputs
        temp = self.dense1(temp)
        temp = self.dropout(temp)
        temp = self.activation_func(temp)
        temp = self.dense(temp)
        return temp


class RepresentationLayer(torch.nn.Module):
    def __init__(self, type, input_dim, output_dim, hidden_dim, **kwargs) -> None:
        super(RepresentationLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.lt = type
        if type == "Linear":
            self.layer = Linear(input_dim, output_dim)
        elif type == "FC":
            self.layer = FullyConnectedLayer(input_dim, output_dim, hidden_dim, dropout_prob=0.2)
        elif type == "LSTM-left":
            self.layer = torch.nn.LSTM(input_size=input_dim, hidden_size=output_dim, bidirectional=True)
        elif type == "LSTM-right":
            self.layer = torch.nn.LSTM(input_size=input_dim, hidden_size=output_dim, bidirectional=True)
        elif type == "LSTM-bidirectional":
            self.layer = torch.nn.LSTM(input_size=input_dim, hidden_size=output_dim/2, bidirectional=True)
        #cnv1d
        #cnv2d
    
    def forward(self, inputs):
        if self.lt == "Linear":
            return self.layer(inputs)
        elif self.lt == "FC":
            return self.layer(inputs)
        elif self.lt == "LSTM-left":
            return self.layer(inputs)[0][:self.hidden_dim]
        elif self.lt == "LSTM-right":
            return self.layer(inputs)[0][self.hidden_dim:]
        elif self.lt == "LSTM-bidirectional":
            return self.layer(inputs)[0]
        
        