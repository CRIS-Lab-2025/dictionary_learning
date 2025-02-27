import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tabulate import tabulate  
from dictionary import AutoEncoder

# Code to take input from csv and output sparse representation via dictionary learning autoencoder


#Load sentences from CSV file
csv_file = 'sentences.csv'  
df = pd.read_csv(csv_file)
sentences = df['sentence'].tolist()  #get list of sentences

#Load the Pythia model and tokenizer
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
tokenizer.pad_token = tokenizer.eos_token

#------ Get activations with hook -----------

#Define a hook to capture activations from an MLP later
activation_list = []

def hook_fn(module, input, output):
    """Hook function to capture activations from the 4th MLP layer."""
    activation_list.append(output)


#Hooking layer at index 3 (4th layer)
layer_to_hook = model.gpt_neox.layers[3].mlp
hook = layer_to_hook.register_forward_hook(hook_fn)

# Tokenize the batch of sentences and run model on bacth
input_ids_batch = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
output = model(**input_ids_batch)

#Retrieve activations from MLP later
activations = activation_list[0] 
print(f"Activations shape: {activations.shape}")

#Sentence level activations - average accross activations ** may need to change this?
sentence_activations = activations.mean(dim=1)  
print(f"Sentence-level activations shape: {sentence_activations.shape}")  


#----- Load autoencoder w/ pretrained dictionary for mlp layer (hardcoded here as 3) ------

ae = AutoEncoder.from_pretrained(
    "dictionaries/pythia-70m-deduped/mlp_out_layer3/10_32768/ae.pt", 
    map_location=torch.device('cpu')
)

features = ae.encode(sentence_activations) #Get features from extracted activations
print(f"Features shape: {features.shape}")  


#----print some basic summary statistics -------- (Can change this to a better visualization later)

#compute summary stats 
def compute_summary_stats(vector):
    return {
        "Mean": np.mean(vector).item(),
        "Variance": np.var(vector).item(),
        "Max Weight": np.max(vector).item(),
        "Max Position": np.argmax(vector).item()
    }

#Compare original w/ sparse 
original_stats = [compute_summary_stats(sentence_activations[i].detach().numpy()) for i in range(len(sentences))]
sparse_stats = [compute_summary_stats(features[i].detach().numpy()) for i in range(len(sentences))]

table_data = [
    [sentences[i], 
     original_stats[i]["Mean"], original_stats[i]["Variance"], original_stats[i]["Max Weight"], original_stats[i]["Max Position"],
     sparse_stats[i]["Mean"], sparse_stats[i]["Variance"], sparse_stats[i]["Max Weight"], sparse_stats[i]["Max Position"]]
    for i in range(len(sentences))
]

headers = ["Sentence", "Orig Mean", "Orig Var", "Orig Max Weight", "Orig Max Pos", "Sparse Mean", "Sparse Var", "Sparse Max Weight", "Sparse Max Pos"]
table = tabulate(table_data, headers=headers, tablefmt="grid", stralign="center", numalign="center")
print(table)

# --- TBD some visualization? ---


hook.remove()



