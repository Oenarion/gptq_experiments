from datasets import load_dataset
import torch
import time

from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.utils import Perplexity
import matplotlib.pyplot as plt
import numpy as np
import gc
import re
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

