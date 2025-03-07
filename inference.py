import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from dictionary import AutoEncoder


class Pythia70Model():

    def __init__(self, dictionaries) -> None:
        self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m-deduped')
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m-deduped')
        self.tokenizer = tokenizer
        self.dictionaries = dictionaries


    def layerwise_sparse_reps(self, input):
        input_ids = self.tokenizer(input, return_tensors="pt").input_ids
        acts = []
        gpt_neox = self.model.base_model
        rotary_emb = gpt_neox.rotary_emb 

        with torch.no_grad():
            hidden_states = gpt_neox.embed_in(input_ids) 
            acts.append(hidden_states.squeeze(0))
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0)
            position_embeddings = rotary_emb(hidden_states, position_ids)
            for i in range(6):
                layer = gpt_neox.layers[i]
                hidden_states = layer(hidden_states, position_embeddings=position_embeddings)[0] 
                acts.append(hidden_states.squeeze(0))

        sparse_reps = []

        for dict in self.dictionaries:
            sparse_reps.append(dict.encode(acts[i+1]).detach())

        return sparse_reps

def inject_activations(self, custom_acts, hook_ind):
    input_len = custom_acts.shape[1]
    dummy_ids = torch.zeros((1, input_len), dtype=torch.int64)
    acts = []
    gpt_neox = self.model.base_model
    rotary_emb = gpt_neox.rotary_emb 
    with torch.no_grad():
        hidden_states = gpt_neox.embed_in(dummy_ids) 
        position_ids = torch.arange(dummy_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0)
        position_embeddings = rotary_emb(hidden_states, position_ids)
        for i in range(6):
            layer = gpt_neox.layers[i]
            if i == hook_ind:
                acts = []
                hidden_states = custom_acts
                acts.append(hidden_states.squeeze(0))
            
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings)[0] 
            acts.append(hidden_states.squeeze(0))

    return acts


def get_all_hook_points(model):
    hook_points = []
    
    for name, module in model.named_modules():
        hook_points.append(name)

    return hook_points


class InferenceEngiene():

    def __init__(self, model_name) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.hook_points = get_all_hook_points(self.model)
    
    def get_hookpoints(self):
        return self.hook_points

    def get_activations(self, sentences, hook_index):
        """
        Returns activations at the indexed hook point for each token in each sentence in the batch.
        
        Args:
            sentences (list of str): List of input sentences.
            hook_index (int): Index of the hook point in self.hook_points.
        
        Returns:
            torch.Tensor: Activations for each token at the hook point.
        """
        hook_point_name = self.hook_points[hook_index]
        activations = []
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]  # Extract first element if it's a tuple
            activations.append(output.detach())  # Detach tensor

        hook = dict(self.model.named_modules())[hook_point_name].register_forward_hook(hook_fn)
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            _ = self.model(input_ids)
        hook.remove()

        return activations[0], dict(self.model.named_modules())[hook_point_name]  if activations else None

    def tokenize_inputs(self, sentences):
        input_ids_batch = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        sentence_dict = {
        i: [token for token in self.tokenizer.convert_ids_to_tokens(input_ids) if token != "<|endoftext|>"]
        for i, input_ids in enumerate(input_ids_batch["input_ids"])
            }

        return sentence_dict
    
    def generate_tokens(self, sentences, max_len, top_k, top_p):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        output_ids = self.model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask,  # Fix: ensures model ignores padding
            max_length=max_len, 
            do_sample=True, 
            top_k=top_k, 
            top_p=top_p,
            pad_token_id = self.tokenizer.pad_token_id  # Fix: prevents padding issues
        )
        generated_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return generated_outputs
    
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
    
    def get_attention_activations(self, sentences, layer):
        activation_list = []
        def attention_hook_fn(module, input, output):
            """Hook function to capture activations from the attention layer."""
            activation_list.append(output)
        layer_to_hook = self.model.gpt_neox.layers[layer].attention
        attention_hook = layer_to_hook.register_forward_hook(attention_hook_fn)
        input_ids_batch = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            _ = self.model(**input_ids_batch) 
        sentence_activations = {}
        for i, input_ids in enumerate(input_ids_batch["input_ids"]):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            sentence_activations[i] = {}
            activations = activation_list[0][0][i]
            for j, token in enumerate(tokens):
                if token != "<|endoftext|>":
                    sentence_activations[i][token] = activations[j].cpu()
        activation_list.clear()

        return sentence_activations



    








    
            