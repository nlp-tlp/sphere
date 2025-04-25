import numpy as np
import random
import torch
import time
import subprocess
import logging


def list2tuple(l):
    return tuple(list2tuple(x) if type(x) == list else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if type(x) == tuple else x for x in t)


def flatten(l): return sum(map(flatten, l), []
                           ) if isinstance(l, tuple) else [l]


def parse_time():
    return time.strftime("%Y.%m.%d-%H.%M.%S", time.localtime())


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return


def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries


def display_memory_usage():
    '''
    Display memory GPU usage of running processes
    '''
    # nvidia-command to run and query GPU memory usage for all processes
    command1 = "nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits"
    command2 = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"
    command_gpu_name = "nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits"
    # Run the command and capture the output
    output1 = subprocess.check_output(command1, shell=True)
    output2 = subprocess.check_output(command2, shell=True)
    output3 = subprocess.check_output(command_gpu_name, shell=True)

    # Split the output into lines to handle multiple processes
    process_memory_info = output1.decode("utf-8").strip().split('\n')
    total_memory_info = output2.decode("utf-8").strip().split('\n')
    gpu_name = output3.decode("utf-8").strip().split('\n')[0]

    memory_total = float(total_memory_info[0]) / 1024  # Convert MB to GB
    memory_usage = None
    for i, memory in enumerate(process_memory_info):
        memory_usage = float(memory) / 1024
        if memory_usage is not None:
            logging.info(f"GPU name: {gpu_name}")
            logging.info(f"GPU memory(process {i}): \
                    {memory_usage: .2f} GB / {memory_total: .2f} GB")
        else:
            logging.info(f"No GPU memory usage found.")
