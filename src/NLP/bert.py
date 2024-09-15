import torch
from torch import Tensor, LongTensor, FloatTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import tiktoken
from typing import Optional, List, Union, Tuple
import pickle
import io
from transformers import BertTokenizer as new_BertTokenizer
from transformers import BertModel as new_BertModel


@dataclass
class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    intermediate_size: int = 4 * hidden_size
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 512
    hidden_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    vocab_root: str = "/Users/a1/.cache/huggingface/hub/models--bert-base-uncased/blobs/fb140275c155a9c7c5a3b3e0e77a9e839594a938"
    param_root: str = "/Users/a1/.cache/huggingface/hub/models--bert-base-uncased/blobs/68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"


class BertEmbeddings(nn.Module):
    def __init__(self, 
                 config: BertConfig
                 ) -> None:
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, 
                input_ids: Optional[LongTensor], 
                attention_mask: Optional[FloatTensor], 
                token_type_ids: Optional[LongTensor],
                ) -> torch.Tensor:
        
        # seq length
        seq_length = input_ids.size(1)  

        # get position_ids
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # get embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        # normalize
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
    

class BertEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask) -> List[Tensor]:
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertLayer(nn.Module):
    def __init__(self, 
                config: BertConfig
                ) -> None:
        super().__init__()
        self.attention = BertSdpaSelfAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.self = BertSdpaSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


class BertSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertSdpaSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask):
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (hidden_states.size(-1) ** 0.5)
        attention_scores = attention_scores * attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = self.output(context_layer, hidden_states)
        return context_layer
    

class BertIntermediate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = GELUActivation()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    

def GELUActivation():
    return nn.GELU()


class BertOutput(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    

class BertPooler(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooler_input = hidden_states[:, 0]
        pooled_output = self.dense(pooler_input)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    

class BertModel(nn.Module):
    def __init__(self, 
                 config: BertConfig
                 ) -> None:
        
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self,input_ids, attention_mask, token_type_ids, pool=False):
        embedding_output = self.embeddings(input_ids, attention_mask, token_type_ids)
        encoder_output = self.encoder(embedding_output, attention_mask)
        if pool:
            pooler_output = self.pooler(encoder_output[-1])
            return pooler_output
        else:
            return encoder_output[-1]
    


    

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(object):

    def __init__(self, vocab: Union[str, List[str]]):
        if isinstance(vocab, str):
            self.vocab = self.load_vocab(vocab)
        elif isinstance(vocab, list):
            self.vocab = vocab
        self.unk_token = "[UNK]"
        self.cls = "[CLS]"
        self.sep = "[SEP]"
        self.pad = "[PAD]"
        self.tokenizer = WordpieceTokenizer(self.vocab, self.unk_token)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        
    def load_vocab(self, vocab):
        list_vocab = []
        with open(vocab, "r", encoding="utf-8") as reader:
            for line in reader:
                list_vocab.append(line.strip())
        return list_vocab
    
    def __call__(self, text: Union[str, List[str], Tuple[str, str], List[Tuple[str, str]]], max_seq_len=512, requires_cls=True):
        """
        There is some conditions here
        1. text is a str sentenec
        2. text is a list of str with a lot of sentences
        3. text is a list of str with a lot of sentences pairs like (sentence A, sentence B)
        """
        if isinstance(text, str):
            input_ids = self.tokenizer.tokenize(text)
            if requires_cls:
                input_ids = [self.word2idx[self.cls]] + input_ids + [self.word2idx[self.sep]]
            attention_mask = torch.ones(size=(1, len(input_ids), len(input_ids)), requires_grad=False)
            token_type_ids = torch.zeros(size=(1, len(input_ids)), requires_grad=False, dtype=torch.long)
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[:max_seq_len]
                attention_mask = attention_mask[:, :max_seq_len, :max_seq_len]
                token_type_ids = token_type_ids[:, :max_seq_len]
            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
            return input_ids, attention_mask,  token_type_ids
        
        elif isinstance(text, List[str]):
            id_list = []
            max_length = -1
            for each in text:
                input_ids = self.tokenizer.tokenize(each)
                if requires_cls:
                    input_ids = [self.word2idx[self.cls]] + input_ids + [self.word2idx[self.sep]]
                if len(input_ids) > max_seq_len:
                    input_ids = input_ids[:max_seq_len]
                if len(input_ids) > max_length:
                    max_length = len(input_ids)
                id_list.append(input_ids)
            for each in id_list:
                each.extend([self.word2idx[self.unk_token]] * (max_length - len(each)))
            input_ids = torch.tensor(id_list, dtype=torch.long)
            attention_mask = torch.ones(size=(len(text), max_length, max_length), requires_grad=False)
            token_type_ids = torch.zeros(size=(len(text), max_length), requires_grad=False, dtype=torch.long)
            return input_ids, attention_mask,  token_type_ids
        
        elif isinstance(text, List[Tuple[str, str]]):
            id_list = []
            token_type_id_list = []
            max_length = -1
            for each in text:
                first_input_ids = self.tokenizer.tokenize(each[0])
                second_input_ids = self.tokenizer.tokenize(each[1])
                input_ids = first_input_ids + [self.word2idx[self.sep]] + second_input_ids
                token_type_ids = [0] * len(first_input_ids) + [0] + [1] * len(second_input_ids)
                if requires_cls:
                    input_ids = [self.word2idx[self.cls]] + input_ids + [self.word2idx[self.sep]]
                    token_type_ids = [0] + token_type_ids + [1]
                if len(input_ids) > max_seq_len:
                    input_ids = input_ids[:max_seq_len]
                    token_type_ids = token_type_ids[:max_seq_len]
                if len(input_ids) > max_length:
                    max_length = len(input_ids)
                id_list.append(input_ids)
                token_type_id_list.append(token_type_ids)
            for each in id_list:
                each.extend([self.word2idx[self.pad]] * (max_length - len(each)))
            for each in token_type_id_list:
                each.extend([self.word2idx[self.pad]] * (max_length - len(each)))
            input_ids = torch.tensor(id_list, dtype=torch.long)
            attention_mask = torch.ones(size=(len(text), max_length, max_length), requires_grad=False)
            token_type_ids = torch.tensor(token_type_id_list, dtype=torch.long)
            return input_ids, attention_mask,  token_type_ids

        elif isinstance(text, Tuple[str, str]):
            first_input_ids = self.tokenizer.tokenize(text[0])
            second_input_ids = self.tokenizer.tokenize(text[1])
            input_ids = first_input_ids + [self.word2idx[self.sep]] + second_input_ids
            token_type_ids = [0] * len(first_input_ids) + [0] + [1] * len(second_input_ids)
            if requires_cls:
                input_ids = [self.word2idx[self.cls]] + input_ids + [self.word2idx[self.sep]]
                token_type_ids = [0] + token_type_ids + [1]
            attention_mask = torch.ones(size=(1, len(input_ids), len(input_ids)), requires_grad=False)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[:max_seq_len]
                attention_mask = attention_mask[:, :max_seq_len, :max_seq_len]
                token_type_ids = token_type_ids[:, :max_seq_len]
            input_ids = torch.tensor(input_ids, dtpye=torch.long).unsqueeze(0)
            return input_ids, attention_mask,  token_type_ids
        
        else:
            raise NotImplementedError("No Such Input Format")

    def __len__(self):
        return len(self.word2idx)

class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.word2index = {w: i for i, w in enumerate(vocab)}

    def tokenize(self, text):
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.word2index[self.unk_token])
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(self.word2index[cur_substr])
                start = end

            if is_bad:
                output_tokens.append(self.word2index[self.unk_token])
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class BertPrediction(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states
    

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        print(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class Bert(nn.Module):
    def __init__(self, 
                 config: BertConfig
                 ) -> None:
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.tokenizer = BertTokenizer(config.vocab_root)
        self.predictions = BertPrediction(config)
        self.bert.embeddings.word_embeddings.weight = self.predictions.decoder.weight

    def forward(self, text, pool=False):
        input_ids, attention_mask, token_type_ids = self.tokenizer(text)
        output = self.bert(input_ids, attention_mask, token_type_ids, pool=pool)
        if not pool:
            prediction = self.predictions(output)
            return output, prediction
        else:
            return output
        

    @staticmethod
    def from_pretrained(path):
        pretrained_params = torch.load(path)
        model = Bert(BertConfig())
        for key, value in pretrained_params.items():
            if "LayerNorm.gamma" in key:
                key = key.replace("LayerNorm.gamma", "LayerNorm.weight")
            if "LayerNorm.beta" in key:
                key = key.replace("LayerNorm.beta", "LayerNorm.bias")
            if "attention.self" in key:
                key = key.replace("attention.self", "attention")
            if "cls." in key:
                key = key.replace("cls.", "")
            if key not in model.state_dict():
                print("missing param:", key)
                continue
            model.state_dict()[key].copy_(value)
        return model


class BertDecoder(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.vocab = self.load_vocab(config.vocab_root)
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}

    def load_vocab(self, vocab):
        list_vocab = []
        with open(vocab, "r", encoding="utf-8") as reader:
            for line in reader:
                list_vocab.append(line.strip())
        return list_vocab

    def forward(self, prediction):
        id_list = prediction.argmax(dim=-1).tolist()
        id_list = id_list[0]
        output_text = ""
        for idx in id_list:
            text_info = self.idx2word[idx]
            if "##" in text_info:
                output_text += text_info.replace("##", "")
            else:
                output_text += " " + text_info
        return output_text
                

    

if __name__ == "__main__":
    config_ = BertConfig()
    sentence = "hello, my dog is cute"
    model_params = torch.load("/Users/a1/PythonProgram/zmcode/model_config/Bert/pytorch_model.bin")
    model = Bert.from_pretrained("/Users/a1/PythonProgram/zmcode/model_config/Bert/pytorch_model.bin")
    input_text = "hello, my bag [MASK] [MASK] cute and i really like it"
    tokenizer = new_BertTokenizer.from_pretrained("bert-base-uncased")
    # huggingface_model = new_BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    # huggingface_embedding = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    # huggingface_output = huggingface_model(huggingface_embedding['input_ids'], huggingface_embedding['attention_mask'])
    # huggingface_output = huggingface_output.last_hidden_state
    print(tokenizer(sentence))
    print(model.tokenizer(sentence))