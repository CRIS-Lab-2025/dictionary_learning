import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from dictionary import AutoEncoder
import os
from pathlib import Path

class GPTneoX_LayerWrapper():

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
            return GPTneoX_LayerWrapper((self.mod, self.tokenizer), (layer, typ), dictionary)
        else:
            raise NotImplementedError()