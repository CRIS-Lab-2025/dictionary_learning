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
    
    def get_mlp_activations(self, sentences, layer):
        activation_list = []
        def hook_fn(module, input, output):
            """Hook function to capture activations from a specified layer."""
            activation_list.append(output)
        layer_to_hook = self.model.gpt_neox.layers[layer].mlp
        hook = layer_to_hook.register_forward_hook(hook_fn)
        input_ids_batch = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            _ = self.model(**input_ids_batch) 
        sentence_activations = {}

        for i, input_ids in enumerate(input_ids_batch["input_ids"]):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            sentence_activations[i] = {}
            activations = activation_list[0][i]
            for j, token in enumerate(tokens):
                if token != "<|endoftext|>":
                    sentence_activations[i][token] = activations[j].cpu()
        activation_list.clear()

        return sentence_activations
    
    def get_mlp_sparse_activations(self, sentences, layer, dictionary_path):
        ae = AutoEncoder.from_pretrained(
            dictionary_path, 
            map_location=torch.device('cpu')
        )
        act_dict = self.get_mlp_activations(sentences, -1)
        sparse_rep_dict = {}
        for i, _ in enumerate(act_dict):
            sparse_rep_dict[i] = {}
            for token in act_dict[i]:
                sparse_rep_dict[i][token] = {}
                sparse_rep = ae.encode(act_dict[i][token]).detach().cpu()
                indices_of_non_zero = np.nonzero(sparse_rep)
                coeffs = [sparse_rep[ind] for ind in indices_of_non_zero]
                sparse_rep_dict[i][token] = coeffs, indices_of_non_zero

        return sparse_rep_dict








    
            