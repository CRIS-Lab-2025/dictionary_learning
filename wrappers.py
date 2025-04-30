import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dictionary import AutoEncoder
import os
from pathlib import Path
import numpy as np

class GPTneoX_DenseWrapper():

    def __init__(self, mod, layer_info) -> None:
        
        self.model, self.tokenizer = mod
        layer_ind, typ = layer_info
        if typ == 'attn':  
            layer = self.model.base_model.layers[layer_ind].attention
        elif typ == 'mlp':
            layer = self.model.base_model.layers[layer_ind].mlp
        else:
            raise NotImplementedError()
        
        self.layer = layer
        self.act = []
        def hook_fn(module, input, output):
            self.act.append(output)
        self.handle = self.layer.register_forward_hook(hook_fn)

    def batch_activations(self, sentences, tokens='all', tokenized_prior=False):
        self.act.clear() 
        if tokenized_prior:
            input_ids, attention_mask = sentences
            toks = attention_mask.sum(dim=1)
            inputs = {
                "input_ids": input_ids.to(self.model.device),
                "attention_mask": attention_mask.to(self.model.device)
            }
            num_sen = input_ids.shape[0]
        else:
            inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=False)
            toks = inputs["attention_mask"].sum(dim=1)
            inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
            num_sen = len(sentences)

        with torch.no_grad():
            self.model(**inputs)

        acts = torch.stack(self.act).squeeze(0)

        if tokens == 'last':
            return acts[torch.arange(num_sen), toks-1 , :], toks
        elif tokens == 'all':
            return acts, toks
        else:
            raise NotImplementedError()
    
        
    
class ActivationWrapper():

    def __init__(self, mod) -> None:
        self.mod = AutoModelForCausalLM.from_pretrained(mod)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.mod = self.mod.to(device)
        tokenizer = AutoTokenizer.from_pretrained(mod)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.mod.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        self.model_name = mod.split('/')[-1]

    def make_layer_wrapper(self, layer, typ):

        if type(self.mod).__name__.endswith("GPTNeoXForCausalLM"):
            return GPTneoX_DenseWrapper((self.mod, self.tokenizer), (layer, typ))
        else:
            raise NotImplementedError()

    def get_vocab(self):
        return self.tokenizer.get_vocab()
    
    def get_reverse_vocab(self):

        temp = self.get_vocab()
        return {v: k for k, v in temp.items()}


    def tokenize_inputs(self, inputs):
        tokens = self.tokenizer(
            inputs,
            padding=True,           
            truncation=True,
            return_tensors='pt'
        )
        token_list = []
        for ids in tokens['input_ids']:
            token_list.append(self.tokenizer.convert_ids_to_tokens(ids))

        return tokens, token_list
    
    def batch_logits(self, inp, tokens = 'all'):

        inputs = self.tokenizer(inp, return_tensors="pt", padding=True, truncation=False)
        toks = inputs["attention_mask"].sum(dim=1)
        inputs = {key: val.to(self.mod.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.mod(**inputs)
            logits = outputs.logits

        if tokens == 'all':
            return logits
        
        elif tokens == 'last':
            return logits[torch.arange(len(inp)), toks-1 , :]
        
    def generate_next_token(self, inp, num_samples, temp):

        log = self.batch_logits(inp, tokens='last') / temp
        probs = F.softmax(log, dim=-1)
        eps = 1e-12
        clamped = probs.clamp(min=eps)
        entropy = -(clamped * clamped.log()).sum(dim=-1)
        entropy = entropy.unsqueeze(1).expand(-1, num_samples) 

        sampled_token_ids = torch.multinomial(probs, num_samples=num_samples, replacement=True) 
        sampled_probs = torch.gather(probs, dim=1, index=sampled_token_ids)   

        vectorized_map = np.vectorize(self.get_reverse_vocab().get)
        tokens_next = vectorized_map(sampled_token_ids)


        return tokens_next, sampled_probs, entropy
    
    def generate_next_token_top2(self, inp):

        log = self.batch_logits(inp, tokens='last')
        probs = F.softmax(log, dim=-1)

        top2_values, top2_indices = torch.topk(log, 2, dim=-1)
        top2_probs = probs.gather(dim=-1, index=top2_indices)
        top2_probs_sum = top2_probs.sum(dim=-1, keepdim=True)
        top2_probs = top2_probs / top2_probs_sum  

        vectorized_map = np.vectorize(self.get_reverse_vocab().get)
        tokens_next = vectorized_map(top2_indices)

        return tokens_next, top2_probs
    
    def generate_and_prepare(self, inp, num_samples=10, temp=0.5):

        tokens_next, sampled_probs, entropy = self.generate_next_token(inp, num_samples, temp)

        tokenized_og_batch, toks = self.tokenize_inputs(inp)
        sen_ten = []

        for i in range(len(inp)):
            sen = toks[i]
            if '[PAD]' in sen:
                before_pad = sen[:sen.index('[PAD]')]
            else:
                before_pad = sen
            for j in range(num_samples):
                merge = before_pad + [str(tokens_next[i,j])]
                sen_ten.append(self.tokenizer.convert_tokens_to_ids(merge))

        max_len = max(len(seq) for seq in sen_ten)
        padded_inputs = [
            seq + [self.tokenizer.pad_token_id] * (max_len - len(seq)) for seq in sen_ten
        ]

        final_sen = [self.tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in sen_ten]
        sampled_probs = sampled_probs.flatten().tolist()
        entropy = entropy.flatten().tolist()

        input_ids = torch.tensor(padded_inputs)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return (input_ids, attention_mask) , final_sen, sampled_probs, entropy

    def generate_and_prepare_top2(self, inp):

        tokens_next, sampled_probs = self.generate_next_token_top2(inp)

        tokenized_og_batch, toks = self.tokenize_inputs(inp)
        sen_ten = []

        for i in range(len(inp)):
            sen = toks[i]
            if '[PAD]' in sen:
                before_pad = sen[:sen.index('[PAD]')]
            else:
                before_pad = sen
            for j in range(2):
                merge = before_pad + [str(tokens_next[i,j])]
                sen_ten.append(self.tokenizer.convert_tokens_to_ids(merge))

        max_len = max(len(seq) for seq in sen_ten)
        padded_inputs = [
            seq + [self.tokenizer.pad_token_id] * (max_len - len(seq)) for seq in sen_ten
        ]

        final_sen = [self.tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in sen_ten]
        sampled_probs = sampled_probs.flatten().tolist()

        input_ids = torch.tensor(padded_inputs)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return (input_ids, attention_mask) , final_sen, sampled_probs


    def prob_of_generation(self, sens, temperature):
        inputs = self.tokenizer(sens, padding=True, truncation=True, return_tensors="pt")
        toks = inputs["attention_mask"].sum(dim=1)
        input_ids = inputs['input_ids']

        log = self.batch_logits(sens, tokens='all') / temperature
        probs = F.softmax(log, dim=-1)

        return probs, input_ids
        




class GPTneoX_SparseWrapper():

    def __init__(self, mod, layer_info, dictionary) -> None:
        
        self.model, self.tokenizer = mod
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.dictionary = dictionary
        layer_ind, typ = layer_info
        if typ == 'attn':  
            layer = self.model.base_model.layers[layer_ind].attention
        elif typ == 'mlp':
            layer = self.model.base_model.layers[layer_ind].mlp
        else:
            raise NotImplementedError()
        
        self.layer = layer
        self.act = []
        def hook_fn(module, input, output):
            self.act.append(output)
        self.handle = self.layer.register_forward_hook(hook_fn)

    def batch_activations(self, sentences, tokens='all'):
        
        self.act.clear() 
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=False)
        toks = inputs["attention_mask"].sum(dim=1)
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}

        with torch.no_grad():
            self.model(**inputs)

        acts = torch.stack(self.act).squeeze(0)

        if tokens == 'last':
            return acts[torch.arange(len(sentences)), toks-1 , :], toks
        elif tokens == 'all':
            return acts, toks
        else:
            raise NotImplementedError()
    
    def batch_sparse_reps(self, sentences, tokens='all'):
        acts, toks = self.batch_activations(sentences)
        if tokens == 'last':
            return self.dictionary.encode(acts[torch.arange(len(sentences)), toks-1 , :]), toks
        elif tokens == 'all':
            return self.dictionary.encode(acts), toks
        else:
            raise NotImplementedError()


class DictionaryWrapper():

    def __init__(self, mod, dictionary_direc) -> None:
        self.mod = AutoModelForCausalLM.from_pretrained(mod)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.mod = self.mod.to(device)
        tokenizer = AutoTokenizer.from_pretrained(mod)
        self.tokenizer = tokenizer
        self.direc = dictionary_direc
        self.model_name = mod.split('/')[-1]

    def make_layer_wrapper(self, layer, typ, num):
        dir = os.path.join(self.direc, self.model_name, typ + '_out_layer' + str(layer))
        dict_dir = os.path.join(dir, str(next(Path(dir).glob(f"{num}_*"), None)))
        dictionary = AutoEncoder.from_pretrained(os.path.join(dict_dir,'ae.pt'))

        if type(self.mod).__name__.endswith("GPTNeoXForCausalLM"):
            return GPTneoX_SparseWrapper((self.mod, self.tokenizer), (layer, typ), dictionary)
        else:
            raise NotImplementedError()
        
    