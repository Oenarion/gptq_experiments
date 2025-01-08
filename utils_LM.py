import matplotlib.pyplot as plt
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.utils import Perplexity
from datasets import load_dataset
import torch
import numpy as np
from transformers import AutoTokenizer
import os 
import re
import time

def get_wikitext2(tokenizer, nsamples, seqlen):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
        lambda x: len(x["text"]) >= seqlen)

    return [tokenizer(example["text"]) for example in traindata.select(range(nsamples))]


# Function to calculate Perplexity distribution
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

def compute_ppl_distribution_for_all_bits(model_id, list_files):
    ppl_distributions = []
    non_quant_model = GPTQModel.load(model_id, QuantizeConfig(bits=4, group_size=128))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("COMPUTING PERPLEXITY FOR NON QUANTIZED MODEL")
    ppl_distributions.append(calculate_ppl_distribution(non_quant_model, tokenizer))

    for file in list_files:
        mod = file.split('-')[6]
        print(f"COMPUTING PERPLEXITY FOR {mod} BITS")
        quant_path = f"wikiText_models\\{file}"
        model = GPTQModel.load(quant_path, device="cuda")
        ppl_distributions.append(calculate_ppl_distribution(model, tokenizer))
    
    return ppl_distributions

def show_distributions(ppl_distributions, configurations):

    plt.figure(figsize=(8, 5))
    for i in range(len(ppl_distributions)):
        if i in configurations.keys():
            plt.plot(ppl_distributions[i], label=configurations[i], marker="o")

    plt.title("Perplexity Distribution")
    plt.xlabel("Chunk Index")
    plt.ylabel("Average Perplexity")
    plt.legend()
    plt.show()


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
        output = model.generate(**inputs, max_length=len(input_text) + max_length)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        generated_part = generated_text[len(input_text)-1:][:len(ground_truth)]
        # Store the result
        results.append({
            "input_text": input_text,
            "ground_truth": ground_truth,
            "generated_text": generated_part
        })

        counter += 1
        i += 1

    return results


def write_results_to_file(filename, results):
    with open(filename, "w", encoding="utf-8", errors="replace") as f:
            for i, nq_res in enumerate(results):
                f.write(f"\nSample {i+1}:\n")
                f.write(f"Input Text: {nq_res['input_text']}\n")
                f.write(f"Ground Truth: {nq_res['ground_truth']}\n")
                f.write(f"Prediction: {nq_res['generated_text']}\n")


def compute_results_for_all_bits(llama_path, tokenizer, list_files, dataset, input_cutoff = 150, num_samples=5, max_length=150):

    if not os.path.exists("LM_results"):
        os.mkdir("LM_results")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    print("Tokenizer loaded!")

    quant_config = QuantizeConfig(bits=8, group_size=128)
    non_quantized_model = GPTQModel.load(llama_path, quant_config)

    non_quantized_results = evaluate_model_on_wikitext(dataset, tokenizer, non_quantized_model,\
                                                        input_cutoff = input_cutoff, num_samples=num_samples, max_length=max_length)
    filename = f"LM_results\\results_wikiText_base.txt"
    write_results_to_file(filename, non_quantized_results)

    for file in list_files:
        quant_path = f"wikiText_models\\{file}"
        mod = file.split('-')[6]
        quantized_model = GPTQModel.load(quant_path)  # Quantized
        print(f"Evaluating for {mod} model...")
        quantized_results = evaluate_model_on_wikitext(dataset, tokenizer, quantized_model,\
                                                        input_cutoff = input_cutoff, num_samples=num_samples, max_length=max_length)
        filename = f"LM_results\\results_wikiText_{mod}.txt"
        write_results_to_file(filename, quantized_results)


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

def measure_token_generation_speed_for_all_bits(model_id, list_files, group_size = 128):
    quantized_times = {}

    prompt = "Uncovering deep insights begins with"

    non_quant_model = GPTQModel.load(model_id, QuantizeConfig(bits=4, group_size=group_size))
    for name, param in non_quant_model.named_parameters():
        print(f"{name}: {param.dtype}")
        break
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("COMPUTING SPEED FOR NON QUANTIZED MODEL")
    nq_avg_time, nq_tokens_per_second = measure_token_generation_speed(non_quant_model, tokenizer, prompt)

    quantized_times[16] = [nq_avg_time, nq_tokens_per_second]

    for file in list_files:
        quant_path = f"wikiText_models\\{file}"
        mod = file.split('-')[6]
        print(f"COMPUTING PERPLEXITY FOR {mod}")

        model = GPTQModel.load(quant_path)
        for name, param in model.named_parameters():
            print(f"{name}: {param.dtype}")
            break
        avg_time, tokens_per_second = measure_token_generation_speed(model, tokenizer, prompt)
        quantized_times[mod] = [avg_time, tokens_per_second]
    
    return quantized_times

def print_speedup(quantized_times, configurations):
    speedups_full_phrase = []
    speedups_token_gen = []

    speedups_full_phrase.append(1)
    speedups_token_gen.append(1)

    nq_avg_full_phrase, nq_avg_token_gen = quantized_times[16]
    print(f"Non-quantized model average inference time: {nq_avg_full_phrase:.4f} seconds per generation")
    print(f"Tokens per second for non quantized model: {nq_avg_token_gen:.2f}")

    for i in range(1, len(configurations)):
        q_avg_full_phrase, q_avg_token_gen = quantized_times[configurations[i]]

        print(f"CONSIDERING MODEL QUANTIZED WITH {configurations[i]}")
        print(f"model average inference time: {q_avg_full_phrase:.4f} seconds per generation")
        print(f"Speedup is : {nq_avg_full_phrase / q_avg_full_phrase:.2f} times")
        print(f"Tokens per second for non quantized model: {q_avg_token_gen:.2f}")
        speedups_full_phrase.append(nq_avg_full_phrase / q_avg_full_phrase)
        speedups_token_gen.append(q_avg_token_gen / nq_avg_token_gen)

    plt.plot(configurations, speedups_full_phrase, marker='o', color='orange')
    plt.xlabel("Quantization Bits")
    plt.ylabel("Speedup Factor")
    plt.title("Speedup Factor Across Quantization Levels for Full Phrase")
    plt.grid(True)
    plt.show()

    plt.plot(configurations, speedups_token_gen, marker='o', color='orange')
    plt.xlabel("Quantization Bits")
    plt.ylabel("Speedup Factor")
    plt.title("Token Generation Speedup Factor Across Quantization Levels")
    plt.grid(True)
    plt.show()