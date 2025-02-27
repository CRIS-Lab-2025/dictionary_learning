import torch as t
from nnsight import LanguageModel
import random

from og.training import trainSAE
from og.utils import hf_dataset_to_generator
from og.buffer import ActivationBuffer
import json

from og.dictionary import AutoEncoder
from og.trainers.standard import StandardTrainer

# Send all these through command line


def sae_training(device, save_dir, model_name, random_seed, layer, dataset_name, params):
    """Mostly follows dictionary_learning/tests/test_end_to_end.py:test_sae_training.

    NOTE: `dictionary_learning` is meant to be used as a submodule. Thus, to run this test, you need to use `dictionary_learning` as a submodule
    and run the test from the root of the repository using `pytest -s`. Refer to https://github.com/adamkarvonen/dictionary_learning_demo for an example"""
    random.seed(random_seed)
    t.manual_seed(random_seed)

    model = LanguageModel(model_name, dispatch=True, device_map=device)


    print("Model loaded")
    print("Using context length:", params["context_length"]["value"])
    print("Using batch size:", params["llm_batch_size"]["value"])
    print("Using SAE batch size:", params["sae_batch_size"]["value"])
    context_length = params["context_length"]["value"]
    llm_batch_size = params['llm_batch_size']['value']
    sae_batch_size = params['sae_batch_size']['value']
    num_contexts_per_sae_batch = sae_batch_size // context_length

    num_inputs_in_buffer = num_contexts_per_sae_batch * 20

    num_tokens = params['num_tokens']['value']
    print(f"Number of tokens: {num_tokens}")

    # sae training parameters
    sparsity_penalty = 2.0
    expansion_factor = 8

    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train
    # Modified this, might be too many?
    save_steps = [int(i / 10) * steps for i in range(10)]
    warmup_steps = int(0.1 * steps)  # Warmup period at start of training and after each resample
    resample_steps = None

    # We'll only do standard training initially
    # standard sae training parameters
    learning_rate = 3e-4

    submodule = model.gpt_neox.layers[layer]
    submodule_name = f"resid_post_layer_{layer}"
    io = "out"
    activation_dim = model.config.hidden_size

    generator = hf_dataset_to_generator(dataset_name)

    # Defining a new activation buffer that can potentially work for distributed training
    activation_buffer = ActivationBuffer(generator, model, submodule, d_submodule=activation_dim, io=io,
                                                    n_ctxs=num_inputs_in_buffer, ctx_len=context_length,
                                                    refresh_batch_size=llm_batch_size, out_batch_size=sae_batch_size,
                                                    device=device)

    # create the list of configs
    trainer_configs = []
    # We'll only try one of these cases.
    trainer_configs.extend(
        [
            {
                "trainer": StandardTrainer,
                "dict_class": AutoEncoder,
                "activation_dim": activation_dim,
                "dict_size": expansion_factor * activation_dim,
                "lr": learning_rate,
                "l1_penalty": sparsity_penalty,
                "warmup_steps": warmup_steps,
                "sparsity_warmup_steps": None,
                "decay_start": None,
                "steps": steps,
                "resample_steps": resample_steps,
                "seed": random_seed,
                # Go through wandb workings?
                "wandb_name": f"StandardTrainer-{model_name}-{submodule_name}",
                "layer": layer,
                "lm_name": model_name,
                # How is device used downstream?
                "device": device,
                "submodule_name": submodule_name,
            },
        ]
    )

    print(f"len trainer configs: {len(trainer_configs)}")
    output_dir = f"{save_dir}/{submodule_name}"

    trainSAE(
        data=activation_buffer,
        trainer_configs=trainer_configs,
        steps=steps,
        save_steps=save_steps,
        save_dir=output_dir,
        verbose=True,
        use_wandb=True,
        wandb_entity = "chris",
        wandb_project = "dictionary_learning",
        log_steps=int(0.1 * steps),
    )

if __name__ == "__main__":
    # All default params are assuming 24 GB GPU
    # Read from params.json
    params_path = "params.json"
    with open(params_path, "r") as f:
        loaded_params = json.load(f)

    DEVICE = loaded_params["device"]["value"]
    print(f"Using device: {DEVICE}")
    SAVE_DIR = loaded_params["save_dir"]["value"]
    MODEL_NAME = loaded_params["model_name"]["value"]
    RANDOM_SEED = loaded_params["random_seed"]["value"]
    LAYER = loaded_params["layer"]["value"]
    DATASET_NAME = loaded_params["dataset_name"]["value"]
    print(f"Using dataset: {DATASET_NAME}")
    print(f"Using layer: {LAYER}")
    print(f"Using save directory: {SAVE_DIR}")
    print(f"Using model name: {MODEL_NAME}")
    sae_training(DEVICE, SAVE_DIR, MODEL_NAME, RANDOM_SEED, LAYER, DATASET_NAME, loaded_params)
