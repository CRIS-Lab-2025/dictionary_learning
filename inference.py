import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from dictionary import AutoEncoder


class InferenceEngiene():
    
    def __init__(self, model_name) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name) # Assume Tokenzier is model-specific
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

    def tokenize_inputs(self, sentences):
        input_ids_batch = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        sentence_dict = {
        i: [token for token in self.tokenizer.convert_ids_to_tokens(input_ids) if token != "<|endoftext|>"]
        for i, input_ids in enumerate(input_ids_batch["input_ids"])
            }

        return sentence_dict
    
    def get_token_embeddings(self, sentences):
        input_ids_batch = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            initial_embeddings = self.model.gpt_neox.embed_in(input_ids_batch["input_ids"])
        emb_dict = {}
        for i, input_ids in enumerate(input_ids_batch["input_ids"]):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            emb_dict[i] = {}
            for j, token in enumerate(tokens):
                if token != "<|endoftext|>":  # Skip padding tokens
                    emb_dict[i][token] = initial_embeddings[i, j].cpu().numpy() 

        return emb_dict


    
            