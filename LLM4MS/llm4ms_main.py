
#########################################
import pandas as pd
import numpy as np
import torch
from torch.nn.functional import normalize
from llm2vec import LLM2Vec
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel


# Loading base model, along with custom code that enables bidirectional connections in decoder-only LLMs. MNTP LoRA weights are merged into the base model.
tokenizer = AutoTokenizer.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
)
config = AutoConfig.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    use_auth_token="xxxxxxxxxxxxxxxxxxx",
)
model = PeftModel.from_pretrained(
    model,
    "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
    use_auth_token="xxxxxxxxxxxxxxxxxxx",
)
model = model.merge_and_unload()  # This can take several minutes on cpu
# Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
model = PeftModel.from_pretrained(
    model, "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse",  
    use_auth_token="xxxxxxxxxxxxxxxxxxx",
)
# Wrapper for encoding and pooling operations
l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)


# # Use your own LLM4MS model
# l2v = LLM2Vec.from_pretrained(
#     "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
#     peft_model_name_or_path="./model/llama",
#     device_map="cuda" if torch.cuda.is_available() else "cpu",
#     torch_dtype=torch.bfloat16,
#     use_auth_token="xxxxxxxxxxxxxxxxxxx",
# )


# Example text template with peak type annotations
text_template = (
    "Instruction: {instruction} "
    "Key Peaks: {key_peaks_text}"
)

# Filtering function: retain m/z and intensity above threshold
def filter_by_intensity(mz, intensity, threshold):
    combined = list(zip(mz, intensity))
    filtered = [(mz_val, int_val) for mz_val, int_val in combined if int_val >= threshold]
    filtered.sort(key=lambda x: -x[1])  # Sort by intensity descending
    return np.array([x[0] for x in filtered]), np.array([x[1] for x in filtered])

# Extract peaks and categorize them
def extract_key_peaks(mz, intensity, top_n=30):
    peaks = list(zip(mz, intensity))
    peaks.sort(key=lambda x: -x[1])  # Sort by intensity descending
    peaks = peaks[:top_n]  # Keep top_n peaks

    if not peaks:
        return None, [], []

    # Extract base_peak (highest intensity)
    base_peak = peaks[0]
    base_mz = base_peak[0]
    max_mz = max(mz) if mz.size > 0 else None

    # Remaining peaks after base_peak
    remaining_peaks = peaks[1:]

    # Extract top 3 key_peaks (next three highest intensities)
    key_peaks = remaining_peaks[:2]
    key_mz = [peak[0] for peak in key_peaks]

    # Extra peaks are all remaining peaks after key_peaks
    extra_peaks = remaining_peaks[2:]
    extra_mz = [peak[0] for peak in extra_peaks]

    return base_mz, key_mz, extra_mz, max_mz

# Format peaks into the required text format
def format_key_peaks(base_mz, key_mz, extra_mz, max_mz):
    parts = []
    
    # Base peak
    if base_mz is not None:
        parts.append(f"base_peak@{int(base_mz)}")
    
    # Key peaks (max 3)
    key_part = ",".join([f"max_peak@{int(mz)}" for mz in key_mz])
    if key_part:
        parts.append(key_part)
    
    # Extra peaks
    extra_part = ",".join([f"extra_peak@{int(mz)}" for mz in extra_mz])
    if extra_part:
        parts.append(extra_part)
        
    # Max peak
    if max_mz is not None:
        parts.append(f"max_peak@{int(max_mz)}")
    
    return ",".join(parts)

# Convert data to template format
def convert_to_text_template(data, column_mz, column_intensity, column_smiles, instruction, threshold, top_n=20):
    templates = []
    for index, row in data.iterrows():
        mz = np.array(tuple(map(float, eval(row[column_mz]))))
        intensity = np.array(tuple(map(float, eval(row[column_intensity]))))
        smiles = row[column_smiles]
        
        # Filter intensity
        filtered_mz, filtered_intensity = filter_by_intensity(mz, intensity, threshold)
        
        # Extract categorized peaks
        base_mz, key_mz, extra_mz, max_mz = extract_key_peaks(filtered_mz, filtered_intensity, top_n=top_n)
        
        # Format text
        key_peaks_text = format_key_peaks(base_mz, key_mz, extra_mz, max_mz)
        template = text_template.format(
            instruction=instruction,
            key_peaks_text=key_peaks_text
        )
        templates.append(template)
    return templates

# Parameter settings
instruction = "This is a mass spectra: "
threshold = 0.005  
top_n = 30  

# Convert data
query_data = pd.read_csv('./small_dataset/query_data_small.csv')
lib_data = pd.read_csv('./small_dataset/lib_data_small.csv')

query_texts = convert_to_text_template(
    query_data, 'MZS', 'INTENSITYS', 'SMILES', instruction, threshold, top_n=top_n
)
document_texts = convert_to_text_template(
    lib_data, 'MZS', 'INTENSITYS', 'SMILES', instruction, threshold, top_n=top_n
)


# Encode queries and documents
q_texts = query_texts  # Text of queries
d_texts = document_texts  # Text of the documents/library

# Encode using LLM2Vec
q_reps = l2v.encode(q_texts)  # Embedding vectors for queries
d_reps = l2v.encode(d_texts)  # Embedding vectors for documents
np.save('q_reps_ori.npy', q_reps)
np.save('d_reps_ori.npy', d_reps)


# Calculate cosine similarity
q_reps_norm = normalize(q_reps, p=2, dim=1)  # Normalize query vectors
d_reps_norm = normalize(d_reps, p=2, dim=1)  # Normalize document vectors
cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))  # Calculate the cosine similarity matrix

# Compare SMILES and calculate accuracy
def calculate_metrics(query_data, lib_data, cos_sim, k_values=[1, 10, 100]):
    """
    Calculate accuracy and recall rates.
    """
    correct_predictions = 0
    total_queries = len(query_data)
    
    # Initialize recall counters
    recall_counts = {k: 0 for k in k_values}

    for query_idx in range(total_queries):
        # Find similarity scores for the current query with all documents and sort them
        similarity_scores = cos_sim[query_idx]
        sorted_indices = similarity_scores.argsort(descending=True).tolist()

        # Get the SMILES of the query
        query_smiles = query_data.iloc[query_idx]['SMILES']

        # Check the top k most similar documents
        for k in k_values:
            # Get the SMILES of the top k most similar documents
            top_k_indices = sorted_indices[:k]
            top_k_smiles = lib_data.iloc[top_k_indices]['SMILES'].tolist()

            # If the query SMILES is among the top k most similar documents, increment the recall count
            if query_smiles in top_k_smiles:
                recall_counts[k] += 1

        # Calculate accuracy (same as recall@1)
        if query_smiles == lib_data.iloc[sorted_indices[0]]['SMILES']:
            correct_predictions += 1

    accuracy = correct_predictions / total_queries
    recalls = {k: recall_counts[k] / total_queries for k in k_values}

    return accuracy, recalls

# Example usage
accuracy, recalls = calculate_metrics(query_data, lib_data, cos_sim)
print(f"Accuracy: {accuracy}")
for k, recall in recalls.items():
    print(f"Recall@{k}: {recall}")