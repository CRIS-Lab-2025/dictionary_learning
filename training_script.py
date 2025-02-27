import torch as t
from nnsight import LanguageModel
import random

from training import trainSAE
from trainers.standard import StandardTrainer
from trainers.top_k import TopKTrainer, AutoEncoderTopK
from utils import hf_dataset_to_generator, get_nested_folders, load_dictionary
from buffer import ActivationBuffer, DistributedActivationBuffer
from dictionary import AutoEncoder
from evaluation import evaluate

# Assuming a GPU cluster with 4 GPUs. Must be changed.
DEVICE = "cuda:0"
SAVE_DIR = "./train_script_results"
MODEL_NAME = "EleutherAI/pythia-2.8b-deduped"
RANDOM_SEED = 42
# Choosing middle layer. Layer lengths can be found here: https://huggingface.co/EleutherAI/pythia-2.8b-deduped
LAYER = 16
DATASET_NAME = "monology/pile-uncopyrighted"


def sae_training():
    """Mostly follows dictionary_learning/tests/test_end_to_end.py:test_sae_training.

    NOTE: `dictionary_learning` is meant to be used as a submodule. Thus, to run this test, you need to use `dictionary_learning` as a submodule
    and run the test from the root of the repository using `pytest -s`. Refer to https://github.com/adamkarvonen/dictionary_learning_demo for an example"""
    random.seed(RANDOM_SEED)
    t.manual_seed(RANDOM_SEED)

    model = LanguageModel(MODEL_NAME, dispatch=True, device_map=DEVICE)


    context_length = 128
    llm_batch_size = 512  # Fits on a 24GB GPU
    sae_batch_size = 8192
    num_contexts_per_sae_batch = sae_batch_size // context_length

    num_inputs_in_buffer = num_contexts_per_sae_batch * 20

    num_tokens = 10_000_000

    # sae training parameters
    sparsity_penalty = 2.0
    expansion_factor = 8

    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train
    # Modified this, might be too many?
    save_steps = [100 * i for i in range(1, steps // 100 + 1)]
    warmup_steps = 1000  # Warmup period at start of training and after each resample
    resample_steps = None

    # We'll only do standard training initially
    # standard sae training parameters
    learning_rate = 3e-4

    submodule = model.gpt_neox.layers[LAYER]
    submodule_name = f"resid_post_layer_{LAYER}"
    io = "out"
    activation_dim = model.config.hidden_size

    generator = hf_dataset_to_generator(DATASET_NAME)

    # Defining a new activation buffer that can potentially work for distributed training
    activation_buffer = ActivationBuffer(generator, model, submodule, d_submodule=activation_dim, io=io,
                                                    n_ctxs=num_inputs_in_buffer, ctx_len=context_length,
                                                    refresh_batch_size=llm_batch_size, out_batch_size=sae_batch_size,
                                                    device='cuda')

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
                "seed": RANDOM_SEED,
                # Go through wandb workings?
                "wandb_name": f"StandardTrainer-{MODEL_NAME}-{submodule_name}",
                "layer": LAYER,
                "lm_name": MODEL_NAME,
                # How is device used downstream?
                "device": 'cuda',
                "submodule_name": submodule_name,
            },
        ]
    )

    print(f"len trainer configs: {len(trainer_configs)}")
    output_dir = f"{SAVE_DIR}/{submodule_name}"

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
        log_steps=100,
    )
