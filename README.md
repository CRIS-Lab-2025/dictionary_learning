# Dictionary Learning for Transformer Activations

This project extracts activations from a specific MLP layer of the Pythia-70m-deduped model, processes them using a pretrained dictionary (autoencoder), and analyzes the sparse representations.

## Features
- Loads and tokenizes sentences from a CSV file.
- Extracts activations (averaged to get sentence level) from a specified MLP layer of Pythia-70m.
- Uses a pretrained autoencoder (dictionary) to obtain sparse representations.
- Computes and displays summary statistics.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install torch transformers pandas tabulate matplotlib numpy
```

## Usage

1. **Prepare the input sentences**:
   - Place your sentences in `CRIS_lab/sentences.csv` under a column named `sentence`.

2. **Run the script**:
   ```bash
   python get_sparse_rep.py
   ```

3. **Expected Output**:
   - Activations from the selected MLP layer.
   - Sparse representations from the autoencoder.
   - Summary statistics comparing original and sparse features.

## Code Breakdown

### 1. **Loading the Model and Data**
- Loads sentences from `sentences.csv`.
- Initializes the Pythia-70m-deduped model and tokenizer.

### 2. **Extracting Activations**
- Hooks into `mlp_out_layer3` (4th MLP layer, index 3 in `gpt_neox.layers`).
- Runs the model on tokenized input sentences.

### 3. **Applying Dictionary Learning**
- Loads a pretrained autoencoder (`ae.pt`) from `mlp_out_layer3`.
- Encodes activations into sparse features.

### 4. **Analyzing the Results**
- Computes mean, variance, max weight, and max position for original and sparse features.
- Displays the results in a table format.

## File Structure
```
CRIS_lab/
├── dictionaries/
│   ├── pythia-70m-deduped/
│   │   ├── mlp_out_layer3/
│   │   │   ├── 10_32768/
│   │   │   │   ├── ae.pt  # Pretrained autoencoder
│   │   │   │   ├── config.json
├── sentences.csv  # Input sentences
├── script.py  # Main script
```

## Notes
- The autoencoder acts as a learned dictionary for sparsely representing activations.
- Ensure that `mlp_out_layer3/10_32768/ae.pt` is present before running.

## Future Work
- Improve visualization of sparse features.
- Experiment with different dictionaries.
- Apply this method to different layers and/or accross all.

## License
MIT License