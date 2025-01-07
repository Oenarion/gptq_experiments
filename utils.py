from datasets import load_dataset
import torch
import time
from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.utils import Perplexity
import matplotlib.pyplot as plt
import numpy as np
import gc
import re

def load_squad_data():
    print("Loading SQuAD dataset...")
    squad = load_dataset("squad", split="train")
    return squad

@torch.no_grad()
def calculate_avg_ppl(model, tokenizer, dataset_path="wikitext", dataset_name="wikitext-2-raw-v1", split="train"):
    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        split=split,
        text_column="text",
    )

    all = ppl.calculate(n_ctx=512, n_batch=512)

    # Average Perplexity
    avg = sum(all) / len(all)

    return avg


# Function to load and process the wikitext dataset
def get_wikitext2(tokenizer, nsamples, seqlen):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
        lambda x: len(x["text"]) >= seqlen)

    return [tokenizer(example["text"]) for example in traindata.select(range(nsamples))]

def quantize_model(dataset, quant_path, model_id, num_bits, group_size = 128):
    quant_config = QuantizeConfig(bits=num_bits, group_size=group_size)
    model = GPTQModel.load(model_id, quant_config)
    print(f"QUANTIZING WITH {num_bits} bits and group size of {group_size} ")
    model.quantize(dataset)
    print("SAVING MODEL")
    model.save(quant_path)

def quantize_for_all_bits_precision(directory, dataset, model_id, dataset_name, group_size = 128):
    for num_bits in [2, 3, 4, 8]:
        quant_path = f"{directory}\\Llama-3.2-1B-Instruct-gptqmodel-{dataset_name}-{num_bits}bit-{group_size}gs"
        quantize_model(dataset, quant_path, model_id, num_bits, group_size = group_size)
  
        
def preprocess_squad_for_quantization(dataset, tokenizer, max_seq_length=512):
    """
    Preprocess SQuAD dataset for quantization.

    Args:
        dataset: List of raw SQuAD entries.
        tokenizer: Hugging Face tokenizer.
        max_seq_length: Maximum sequence length for tokenization.

    Returns:
        Preprocessed dataset with input_ids and attention_mask.
    """
    preprocessed_data = []
    for item in dataset:
        question = item["question"]
        context = item["context"]
        
        # Tokenize the input pair (question, context)
        tokenized = tokenizer(
            question,
            context,
            max_length=max_seq_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        # Append processed data with input_ids and attention_mask
        preprocessed_data.append({
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
        })

    return preprocessed_data


def preprocess_squad_for_evaluation(dataset, tokenizer, max_seq_length=512):
    preprocessed_data = []
    for item in dataset:
        question = item.get("question")
        context = item.get("context")
        answers = item.get("answers", {}).get("text", [])  # Default to an empty list if no answers

        # Tokenize the question and context
        tokenized = tokenizer(
            question,
            context,
            max_length=max_seq_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        # Find the token positions of the answer in the context
        answer_start = item.get("answers", {}).get("answer_start", [])
        start_positions = []
        end_positions = []

        for i, answer in enumerate(answers):
            # answer_start[i] should be the starting position for each answer
            answer_end = answer_start[i] + len(answer)  # Calculate the end position of the answer
            
            # Convert char indices to token indices
            start_token_idx = tokenized.char_to_token(answer_start[i])
            end_token_idx = tokenized.char_to_token(answer_end - 1)  # -1 because end is exclusive
            
            if start_token_idx is not None and end_token_idx is not None:
                start_positions.append(start_token_idx)
                end_positions.append(end_token_idx)
        
        # If no valid positions found (this can happen if the tokenizer splits the answer)
        if not start_positions or not end_positions:
            start_positions = [0]  # Default to 0, it means no valid span found
            end_positions = [0]

        preprocessed_data.append({
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "question": question,
            "context": context,
            "answers": {"text": answers},
            "start_positions": start_positions[0],  # Use the first valid answer span
            "end_positions": end_positions[0],      # Use the first valid answer span
        })

    return preprocessed_data


def measure_inference_time(model, tokenizer, prompt, num_runs=10):
    start_time = time.time()
    for _ in range(num_runs):
        result = model.generate(
            **tokenizer(prompt, return_tensors="pt").to(model.device)
        )[0]
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    return avg_time


def evaluate_model_on_wikitext(test_data, tokenizer, model, input_cutoff = 150, num_samples=5, max_length=150):
    results = []
    i, counter = 0, 0
    while counter < num_samples:
        # Select a random example from the test set
        sample = test_data[i]["text"]

        # I only want lengthy samples to evaluate model's performance
        if len(sample) < 200:
            i += 1
            continue

        if len(sample) > input_cutoff:
            # Match up to the nearest word boundary after the cutoff
            match = re.match(r"^(.{100,}?\b)", sample[:170])  
            if match:
                input_text = match.group(1)
            else:
                continue  # Fallback in case regex fails
        else:
            continue

        # Use the remaining text for the ground truth
        ground_truth_start = len(input_text)
        ground_truth = sample[ground_truth_start:ground_truth_start + max_length]

        # Generate a response
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_length=max_length)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Store the result
        results.append({
            "input_text": input_text,
            "ground_truth": ground_truth,
            "generated_text": generated_text
        })

        counter += 1
        i += 1

    return results

@torch.no_grad()
def calculate_ppl_distribution(model, tokenizer, dataset_path="wikitext", dataset_name="wikitext-2-raw-v1", split="test", n_chunks=10):
    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        split=split,
        text_column="text",
    )

    all_ppl = ppl.calculate(n_ctx=512, n_batch=512)
    chunk_size = len(all_ppl) // n_chunks
    distributions = [np.mean(all_ppl[i:i + chunk_size]) for i in range(0, len(all_ppl), chunk_size)]
    return distributions

def compute_ppl_distribution_for_all_bits(model_id, group_size = 128):
    ppl_distributions = []
    non_quant_model = GPTQModel.load(model_id, QuantizeConfig(bits=4, group_size=128))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("COMPUTING PERPLEXITY FOR NON QUANTIZED MODEL")
    ppl_distributions.append(calculate_ppl_distribution(non_quant_model, tokenizer))

    for num_bits in [2, 3, 4, 8]:
        print(f"COMPUTING PERPLEXITY FOR {num_bits} BITS")
        quant_path = f"Llama-3.2-1B-Instruct-gptqmodel-{num_bits}bit-{group_size}gs"
        model = GPTQModel.load(quant_path)
        ppl_distributions.append(calculate_ppl_distribution(model, tokenizer))

    return ppl_distributions


def measure_token_generation_speed(model, tokenizer, prompt, num_runs=10):
    total_tokens = 0
    start_time = time.time()
    for _ in range(num_runs):
        result = model.generate(
            **tokenizer(prompt, return_tensors="pt").to(model.device)
        )
        total_tokens += result.shape[-1]  # Number of tokens in the output
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    tokens_per_second = total_tokens / (end_time - start_time)
    return avg_time, tokens_per_second

def measure_token_generation_speed_for_all_bits(model_id, group_size = 128):
    avg_times = []
    tokens_per_seconds = []
    
    prompt = "Uncovering deep insights begins with"

    non_quant_model = GPTQModel.load(model_id, QuantizeConfig(bits=4, group_size=group_size))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("COMPUTING SPEED FOR NON QUANTIZED MODEL")
    nq_avg_time, nq_tokens_per_second = measure_token_generation_speed(non_quant_model, tokenizer, prompt)

    for num_bits in [8, 4, 3, 2]:
        print(f"COMPUTING PERPLEXITY FOR {num_bits} BITS")
        quant_path = f"Llama-3.2-1B-Instruct-gptqmodel-{num_bits}bit-{group_size}gs"
        model = GPTQModel.load(quant_path)
        avg_time, tokens_per_second = measure_token_generation_speed(model, tokenizer, prompt)
        avg_times.append(avg_time)
        tokens_per_seconds.append(tokens_per_second)
    
    
    return nq_avg_time, nq_tokens_per_second, avg_times, tokens_per_seconds