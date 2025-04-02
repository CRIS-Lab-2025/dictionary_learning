import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dictionary import AutoEncoder
import gc


class Pythia_model:
    def __init__(self, model_path, tokenizer_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()
        self.device = device
        # Get the sparse encoder trained on the 5th layer.
        self.sae = AutoEncoder.from_pretrained(
            "dictionaries/pythia-70m-deduped/mlp_out_layer5/10_32768/ae.pt",
            map_location=device
        )

    def tokenize(self, text):
        # print("getting inputs ids")
        return self.tokenizer(text, return_tensors="pt").input_ids

    # Need to include batch size dimension to the tokenized inputs.
    def get_final_layer_activations(self, tokenized_input):
        with torch.no_grad():
            outputs = self.model(tokenized_input, output_hidden_states=True)
            # Is it that simple? test it out with an example
            return outputs.hidden_states[-1]

    def get_activation_for_token(self, sentence, keyword):
        # Identify the position of the keyword in the tokenized input
        # We need to convert the tensor to the token string
        tokenized_input = self.tokenize(sentence)
        tokenized_input = tokenized_input.to(self.device)
        # print("tokenized input", tokenized_input)
        position = self.get_keyword_position(sentence, keyword)
        if position == -1:
            return None

        try:
            final_layer_activations = self.get_final_layer_activations(tokenized_input)
        except:
            print("Error in getting final layer activations", len(tokenized_input[0]))
            return None
        # clean up code
        finally:
            del tokenized_input
            gc.collect()
            torch.cuda.empty_cache()
        return final_layer_activations[0][position]

    def get_keyword_position(self, sentence, keyword):
        # Identify the position of the keyword in the tokenized input
        # We need to convert the tensor to the token string
        tokenized_input = self.tokenize(sentence)
        tokenized_input = tokenized_input.to(self.device)
        # print("tokenized input", tokenized_input)
        keyword = keyword.lower()
        # Get the positions corresponding to the keywords
        decoded_strings = self.tokenizer.convert_ids_to_tokens(tokenized_input[0])
        # print("decoded strings", decoded_strings)
        position = -1
        for i in range(len(decoded_strings)):
            if keyword in decoded_strings[i]:
                # print("Found the position of the keyword", keyword, "at position", i)
                position = i
        return position

    def get_sparse_activations_for_token(self, sentence, keyword):
        # Identify the position of the keyword in the tokenized input
        # We need to convert the tensor to the token string
        activations = self.get_activation_for_token(sentence, keyword)
        if activations is None:
            return None
        # Get the sparse rep of these tokens
        # Add batch dimension
        print("activations device", activations.device)
        activations = activations.unsqueeze(0)
        # Initialize the sparse autoencoder model for the final layer.
        sparse_acts = self.sae.encode(activations)
        return sparse_acts