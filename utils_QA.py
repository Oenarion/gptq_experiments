import random
import torch
import numpy as np
import re
import time
from collections import Counter
import string
import matplotlib.pyplot as plt
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset



def load_squad_data():
    print("Loading SQuAD dataset...")
    squad = load_dataset("squad")
    return squad

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_answer_of_model(predicted_answer):
    # Look for the "Answer:" keyword
    answer_start_idx = predicted_answer.find("Answer:")
    if answer_start_idx != -1:
        # Extract the part after "Answer:"
        answer_part = predicted_answer[answer_start_idx + len("Answer:"):].strip()
        
        # Look for the "Explanation:" keyword within the extracted part
        explanation_start_idx = answer_part.find("Explanation:")
        if explanation_start_idx != -1:
            # Return only the part before "Explanation:"
            return answer_part[:explanation_start_idx].strip()
        
        # Return everything after answer if no explanation
        return answer_part.strip()
    
    # If answer is not found return everything
    return predicted_answer.strip()


def zero_shot_qa_no_deterministic(tokenizer, context, question, model):
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Send inputs to GPU
    outputs = model.generate(**inputs, max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = get_answer_of_model(answer)
    return answer

def zero_shot_qa(tokenizer, context, question, model):
    prompt = f"""
    Context: {context}

    Question: {question}

    Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        # generated max 10 tokens to evaluate speedup
        outputs = model.generate(**inputs, max_new_tokens=10, 
                                 eos_token_id=tokenizer.eos_token_id, 
                                 do_sample=False, 
                                 num_beams=1,
                                 temperature=None, # Remove irrelevant parameter
                                 top_p=None )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = get_answer_of_model(answer)
    return outputs, answer

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punctuation(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def compute_f1(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        # If either is empty, return 0 F1
        return 0.0
    
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Function to calculate Exact Match
def compute_exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def save_results_to_file(filename, results, em_scores, f1_scores, avg_times, avg_speeds, avg_tokens):
    with open(filename, "w", encoding="utf-8", errors="replace") as f:
        for idx, result in enumerate(results):
            # print(f"Problematic prediction: {result['predicted']}")
            f.write(f"Example {idx + 1}\n")
            f.write(f"Question: {result['question']}\n")
            f.write(f"Ground Truth: {result['ground_truth']}\n")
            f.write(f"Predicted: {result['predicted']}\n\n")
            f.write(f"Exact Match: {result['em']}\n")
            f.write(f"F1 Score: {result['f1']:.2f}\n")
            f.write(f"Time to generate response: {result['time']}s\n\n")
        
        f.write(f"Average Exact Match (EM): {sum(em_scores) / len(em_scores):.2f}\n")
        f.write(f"Average F1 Score: {sum(f1_scores) / len(f1_scores):.2f}\n")
        f.write(f"Average Time for response generation: {sum(avg_times) / len(avg_times):.2f}s\n")
        f.write(f"Average Number of tokens generated: {sum(avg_tokens) / len(avg_tokens):.2f}\n")
        f.write(f"Average Tokens generated each second: {sum(avg_speeds) / len(avg_speeds)}")

def compute_results(tokenizer, squad_evaluation_dataset, filename, model):
    em_scores = []
    f1_scores = []
    results = []
    avg_times = []
    avg_tokens = []
    avg_speeds = []
    
    for example in squad_evaluation_dataset:
        context = example["context"]
        question = example["question"]
        ground_truth = example["answers"]["text"][0]
        prompt = f"""
        Context: {context}

        Question: {question}

        Answer:"""
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        for _ in range(5):  # Run for warm-up
            _ = model.generate(**inputs)
        
        times = []
        token_counts = []
    
        for _ in range(5):
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete
            start_time = time.perf_counter()
            
            outputs, predicted_answer = zero_shot_qa(tokenizer, context, question, model)
            
            torch.cuda.synchronize()  # Ensure generation is complete
            end_time = time.perf_counter()
            curr_time = (end_time - start_time)
            times.append(curr_time)
            token_counts.append(outputs.shape[1] - inputs["input_ids"].shape[1])

        # Compute metrics
        em = compute_exact_match(predicted_answer, ground_truth)
        f1 = compute_f1(predicted_answer, ground_truth)
        
        avg_time = sum(times) / len(times)
        avg_times.append(avg_time)
        
        avg_token = sum(token_counts) / len(token_counts)
        avg_tokens.append(avg_token)
        
        avg_speed = avg_token / avg_time
        avg_speeds.append(avg_speed)

        em_scores.append(em)
        f1_scores.append(f1)
        results.append({"question": question, "ground_truth": ground_truth, 
                        "predicted": predicted_answer, "em": em, "f1": f1, "time": avg_time})

    save_results_to_file(filename, results, em_scores, f1_scores, avg_times, avg_speeds, avg_tokens)
    return avg_times, avg_tokens, avg_speeds

def print_speedup(quantized_times, configurations):
    
    speedups_full_phrase = []
    # speedups_token_gen = []
    
    speedups_full_phrase.append(1)
    # speedups_token_gen.append(1)
    
    nq_avg_full_phrase, _, nq_avg_token_gen = quantized_times[32]
    
    print(f"avg non quantized full phrase: {nq_avg_full_phrase}")
    print(f"avg non quantized token gen: {nq_avg_token_gen}")

    for i in range(1,len(configurations)):
        q_avg_full_phrase, _, q_avg_token_gen = quantized_times[configurations[i]]
        
        print(f"CONSIDERING MODEL QUANTIZED WITH {configurations[i]} bits")
        print(f"model average inference time: {q_avg_full_phrase:.4f} seconds per generation")
        print(f"Speedup for full phrase is : {nq_avg_full_phrase / q_avg_full_phrase:.2f} times")
        speedups_full_phrase.append(nq_avg_full_phrase / q_avg_full_phrase)
        
        print(f"model average token generation speed: {q_avg_token_gen:.4f}")
        print(f"Token Generation Speedup Factor: {q_avg_token_gen / nq_avg_token_gen:.2f} times")
        # speedups_token_gen.append(q_avg_token_gen / nq_avg_token_gen)

    plt.plot(configurations, speedups_full_phrase, marker='o', color='orange')
    plt.xlabel("Quantization Bits")
    plt.ylabel("Speedup Factor")
    plt.title("Speedup Factor Across Quantization Levels for Full Phrase")
    plt.grid(True)
    plt.show()

    # plt.plot(configurations, speedups_token_gen, marker='o', color='orange')
    # plt.xlabel("Quantization Bits")
    # plt.ylabel("Speedup Factor")
    # plt.title("Token Generation Speedup Factor Across Quantization Levels")
    # plt.grid(True)
    # plt.show()

    #return speedups_full_phrase, speedups_token_gen

def save_results_for_all_models(tokenizer, dataset, llama_path, directory):
    if not os.path.exists("QA_results"):
        os.mkdir("QA_results")
    
    # tokenizer = AutoTokenizer.from_pretrained(llama_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    print("Tokenizer loaded!")
    # evaluate for non quantized model first
    quant_config = QuantizeConfig(bits=8, group_size=128)
    model = GPTQModel.load(llama_path, quant_config)
    # model = model.to(torch.float16)
    model = model.to("cuda")
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('MODEL SIZE: {:.3f}MB'.format(size_all_mb))
    model.eval()
    print("Base model loaded!")
    
    filename = "QA_results\\results_SQuAD_base.txt"
    quantized_times = {}
    
    print("Computing results for base model...")
    nq_avg_times, nq_avg_tokens, nq_avg_speeds = compute_results(tokenizer, dataset, filename, model)
    quantized_times[32] = [np.mean(nq_avg_times), np.mean(nq_avg_tokens), np.mean(nq_avg_speeds)]
    print("...End!")
    
    for file in directory:
        # quant_path = f"Llama-3.2-1B-Instruct-gptqmodel-{dataset_name}-{num_bits}bit-{group_size}gs"
        quant_path = f"SQuAD_models\\{file}"
        mod = file.split('-')[6]
        model = GPTQModel.from_quantized(quant_path, device="cuda")
        # model = model.to(torch.float16)
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print('MODEL SIZE: {:.3f}MB'.format(size_all_mb))
        model.eval()
        print(f"Loaded quantized {mod} bit model")
        filename = f"QA_results\\results_SQuAD_quantized_{mod}.txt"
        print("Computing results for quantized model...")
        q_avg_times, q_avg_tokens, q_avg_speeds = compute_results(tokenizer, dataset, filename, model)
        quantized_times[mod] = [np.mean(q_avg_times), np.mean(q_avg_tokens), np.mean(q_avg_speeds)]
        print("...End!")
    
    return quantized_times
