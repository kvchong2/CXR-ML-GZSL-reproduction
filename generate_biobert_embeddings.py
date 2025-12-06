# Generate BioBERT embeddings for NIH Chest X-ray disease class names.
#
# This script creates embeddings for the 14 disease classes used in the CheXNet training.
# The embeddings are saved in the same format as the pre-computed embeddings file
# (embeddings/nih_chest_xray_biobert.npy) for use in train_chexnet.ipynb.
#
# The class names are extracted from the dataset definition and match the order
# used in the training code.

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
import pandas as pd

# Generate BioBERT embeddings for a list of texts
# Returns numpy array of shape [num_texts, embedding_dim] containing the embeddings
def get_biobert_embeddings(texts, model_name='dmis-lab/biobert-v1.1'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i, text in enumerate(texts):
            # Tokenize the text
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )

            # Get model outputs
            outputs = model(**inputs)

            # Extract [CLS] token embedding (first token) for single-word labels
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

            embeddings.append(embedding)

    # Stack into a numpy array
    embeddings_array = np.stack(embeddings)

    print(f"Generated embeddings with shape: {embeddings_array.shape}")
    return embeddings_array

if __name__ == '__main__':
    # Define the 14 disease classes in the same order as in dataset.py
    # This order must match the order used in the training code
    # Note: The order is important and must match the dataset class order
    CLASSES = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening',
        'Hernia'
    ]

    # Generate embeddings using [CLS] token (appropriate for single-word disease labels)
    embeddings = get_biobert_embeddings(CLASSES)

    # Create output directory if it doesn't exist
    output_dir = 'embeddings'
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings
    output_path = os.path.join(output_dir, 'nih_chest_xray_biobert.npy')
    np.save(output_path, embeddings)

    print(f"Successfully saved embeddings to: {output_path}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
