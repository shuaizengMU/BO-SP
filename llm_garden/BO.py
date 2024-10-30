import os
import torch
import numpy as np
import pandas as pd
# import esm
import esm_adapter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set directory for torch hub
# torch.hub.set_dir("/home/nmnx/nmn_data/torch_hub")


dtype = torch.double

SMOKE_TEST = os.environ.get("SMOKE_TEST")
N_TRIALS = 3 if not SMOKE_TEST else 2
N_BATCH = 20 if not SMOKE_TEST else 2
MC_SAMPLES = 256 if not SMOKE_TEST else 32

##########################################################################################
from typing import List, Tuple

def get_esm_example_data(len=None) -> List[Tuple[str, str]]:
    """
    Returns example data for ESM.
    """
    data = [
        ('protein1', 'KVNTIPNGALNS'),
        ('protein2', 'YSTTGHVDYVGR'),
        ('protein3', 'AGTGVHMRLGGL'),
        ('protein4', 'IPLVDTDYMTSR'),
        ('protein5', 'VIAKHTRPVMAG'),
        ('protein6', 'AHGHTPMLTDSM'),
        ('protein7', 'GYRHVPMSPPMG'),
        ('protein8', 'GHKHDNMSMESL'),
        ('protein9', 'GHKHTFEITPRA'),
        ('protein10', 'SHVHVHGGYKQK'),
        ('protein11', 'GHKHVSLPPWVE'),
        ('protein12', 'NHKHIQYPRYNA'),
        ('protein13', 'LPVSHPHRMQTD'),
        ('protein14', 'IVTVGMPLTGPK'),
        ('protein15', 'WHWWPWVDSQNT'),
        ('protein16', 'HLPTWVMWPLSA'),
        ('protein17', 'WHWTWFSQAMMA')
    ]

    if len is not None:
        data = [(one_data[0], one_data[1][:len]) for one_data in data]
    
    return data

INPUT_SEQ_SIZE = 20
data = get_esm_example_data(INPUT_SEQ_SIZE)

# Load ESM-2 model
model, alphabet = esm_adapter.pretrained.esm2_t6_8M_UR50D()

batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

batch_labels, batch_strs, batch_tokens = batch_converter(data)

if torch.cuda.is_available():
    batch_tokens = batch_tokens.to(device='cuda')

# Print input data and shape
data_trun = [(label, seq[:INPUT_SEQ_SIZE]) for label, seq in data]
print(data_trun)

# Get the output representation from the model
esm_output_dict = model(batch_tokens, repr_layers=[model.num_layers])
representation = esm_output_dict["representations"][model.num_layers]

# Print the original representation tensor shape
print(f"Original representation shape: {representation.size()}")

# Filter out the first (beginning <cls>) and last (ending <eos>) token embeddings
filtered_representation = representation[:, 1:-1, :]  # Exclude the first and last tokens

# Print the filtered representation tensor shape
print(f"Filtered representation shape: {filtered_representation.size()}")

########################################################################################

# Token embedding for completeness, but not needed if you are only using `representation`
token_embedding = model.embed_tokens(batch_tokens)

# Print token embeddings and shape for reference
print("Token embedding for each protein sequence:", token_embedding)
print("Token embedding shape:", token_embedding.shape)

# Assign grades to each protein sequence
grades = {
    'protein1': 0.5,
    'protein2': 2.7,
    'protein3': 1.2,
    'protein4': 0.7,
    'protein5': 1.7,
    'protein6': 2.0,
    'protein7': 2.7,
    'protein8': 3.2,
    'protein9': 3.0,
    'protein10': 4.7,
    'protein11': 4.0,
    'protein12': 3.7,
    'protein13': 1.0,
    'protein14': 1.6,
    'protein15': 3.3,
    'protein16': 3.0,
    'protein17': 2.0
}

# Extract grades corresponding to the truncated sequences
targets_pre = torch.tensor([grades[label] for label, _ in data_trun])

print("Grades for each protein sequence:", targets_pre)


import pandas as pd
import torch

flattened_embeddings = filtered_representation.cpu().detach().numpy().reshape(filtered_representation.shape[0], -1)

sequences = [seq for _, seq in data_trun]

labels = targets_pre.tolist()  # Convert tensor to a list

df_embeddings = pd.DataFrame(flattened_embeddings)

df_embeddings['sequence'] = sequences
df_embeddings['labels'] = labels

csv_file_path = r"C:\Users\nmn5x\Desktop\BO\esm_bo\data\data_gary_31_filt_reps_pca_acq.csv"

df_embeddings.to_csv(csv_file_path, index=False)

print(f"Filtered data saved to {csv_file_path}")

from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from sklearn.decomposition import PCA
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
import esm  # Importing esm for the model

# Load the filtered dataset
file_path = r'C:\Users\nmn5x\Desktop\BO\esm_bo\data\data_gary_31_filt_reps_pca_acq.csv'
data = pd.read_csv(file_path)

X = data.iloc[:, :-2].values  # Exclude the last two columns which are 'sequence' and 'labels'
y = data['labels'].values

pca = PCA(n_components=16)  # Retain 95% variance
X_reduced = pca.fit_transform(X)

print(f"Original feature size: {X.shape[1]}")
print(f"Reduced feature size: {X_reduced.shape[1]}")

X_train = X_reduced
y_train = y

train_X = torch.tensor(X_train, dtype=torch.double)
train_Y = torch.tensor(y_train, dtype=torch.double).unsqueeze(-1)  # Ensure Y is a column vector
print("train_Y =", train_Y)

train_X_standardized = (train_X - train_X.mean(dim=0)) / train_X.std(dim=0)

# Apply Min-Max scaling to ensure data is within [0, 1]
train_X_min = train_X_standardized.min(dim=0, keepdim=True).values
train_X_max = train_X_standardized.max(dim=0, keepdim=True).values
train_X_normalized = (train_X_standardized - train_X_min) / (train_X_max - train_X_min + 1e-8)  # Avoid division by zero

train_Y_mean = train_Y.mean()
train_Y_std = train_Y.std()
train_Y_standardized = (train_Y - train_Y_mean) / train_Y_std

all_X = train_X_normalized
all_Y = train_Y_standardized

max_iterations = 10
new_points_normalized = []

for iteration in range(max_iterations):
    print(f"--- Iteration {iteration + 1} ---")

    model_GP = SingleTaskGP(all_X, all_Y)

    mll = ExactMarginalLogLikelihood(model_GP.likelihood, model_GP)

    # Fit the model using the MLL
    fit_gpytorch_model(mll)

    ucb = UpperConfidenceBound(model=model_GP, beta=0.2)  # Adjust beta for more exploration

    # Optimize acquisition function to find the next point
    bounds = torch.stack([torch.zeros(all_X.shape[1]), torch.ones(all_X.shape[1])])
    candidate, _ = optimize_acqf(ucb, bounds=bounds, q=1, num_restarts=5, raw_samples=20)

    # Store the normalized candidate point
    new_points_normalized.append(candidate)

    model_GP.eval()
    with torch.no_grad():
        posterior = model_GP(candidate)
        pred_mean = posterior.mean
        pred_uncertainty = posterior.variance.sqrt()

    pred_mean_original = pred_mean * train_Y_std + train_Y_mean
    pred_uncertainty_original = pred_uncertainty * train_Y_std  # Only scaled by std

    print(f"Next point (normalized): {candidate}")
    print(f"Predicted mean: {pred_mean_original.item()}")
    print(f"Predicted uncertainty (std dev): {pred_uncertainty_original.item()}")

    new_y = torch.tensor([[pred_mean_original.item()]], dtype=torch.double)  # Simulating label
    all_X = torch.cat([all_X, candidate], dim=0)
    all_Y = torch.cat([all_Y, (new_y - train_Y_mean) / train_Y_std], dim=0)  # Standardized new_y

    print(f"Updated dataset size: {all_X.size(0)}")

print("Bayesian Optimization loop finished.")

esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()

# After the optimization loop, decode the new points back to original feature size
pred_seqs = []
for idx, new_point_normalized in enumerate(new_points_normalized):
    # Step 1: Rescale from [0, 1] back to the original standardized scale
    new_point_standardized = new_point_normalized * (train_X_max - train_X_min + 1e-8) + train_X_min

    # Step 2: Reverse standardization to get back to the original data scale
    new_point_original = new_point_standardized * train_X.std(dim=0) + train_X.mean(dim=0)

    # Step 3: Inverse PCA to map back to the original feature space (3840 dimensions)
    new_point_full_representation = pca.inverse_transform(new_point_original.cpu().detach().numpy())

    new_point_full_representation_torch = torch.tensor(new_point_full_representation, dtype=torch.float).view(1, 12, 320)  # Assuming (batch_size, seq_len, hidden_size)

    esm_model.eval()
    with torch.no_grad():
        logits = esm_model.lm_head(new_point_full_representation_torch)

    tokens = torch.argmax(logits, dim=2)
    decoded_sequence = "".join([alphabet.get_tok(tok.item()) for tok in tokens[0]])  # No need to remove CLS and EOS as they're already removed

    pred_seqs.append(decoded_sequence)
    print(f"Decoded sequence for iteration {idx + 1}: {decoded_sequence}")


