import os
import torch
import atexit
import datetime
from collections import defaultdict
from megatron.core import parallel_state


LOGGED_TENSORS = []
LOGGED_GRADS = defaultdict(list)
local_micro_batch_id = -1


RANK = os.environ.get("RANK", "0")
WORLD_SIZE = os.environ.get("WORLD_SIZE", "1")
parallel_state_initialized = False
DP_RANK = DP_SIZE = PP_RANK = TP_RANK = None


logging_tensor = lambda name, tensor: None
logging_grad = lambda name, tensor, iteration: None


def _init_ranks():
    global DP_RANK, DP_SIZE, PP_RANK, TP_RANK
    DP_RANK = parallel_state.get_data_parallel_rank()
    DP_SIZE = parallel_state.get_data_parallel_world_size()
    PP_RANK = parallel_state.get_pipeline_model_parallel_rank()
    TP_RANK = parallel_state.get_tensor_model_parallel_rank()


def _logging_tensor(name, tensor):
    if tensor is None:
        return
    global LOGGED_TENSORS, parallel_state_initialized, local_micro_batch_id, DP_RANK, DP_SIZE
    if not parallel_state_initialized:
        _init_ranks()
        parallel_state_initialized = True
    if name == "tokens":
        local_micro_batch_id += 1
    glb_batch_id = local_micro_batch_id * DP_SIZE + DP_RANK
    LOGGED_TENSORS.append((f"batch_{glb_batch_id}.{name}", tensor.detach().cpu()))


def _logging_grad(name, tensor, iteration, require_tpdp=False):
    # rely on the hook to have the tp rank and pp rank in name.  Cannot go beyond one minibatch. has no index for it to tell the diff.
    if tensor is None:
        return
    global LOGGED_GRADS, parallel_state_initialized
    if not parallel_state_initialized:
        _init_ranks()
        parallel_state_initialized = True
    if require_tpdp:
        name = f"tp_{TP_RANK}.pp_{PP_RANK}.{name}"
    LOGGED_GRADS[iteration].append((f"{name}", tensor.detach().cpu()))


def remove_grad(iteration):
    global LOGGED_GRADS
    if iteration in LOGGED_GRADS:
        del LOGGED_GRADS[iteration]


def save_logged_tensors(rank, world_size, date):
    global LOGGED_TENSORS
    log_path_fwd = os.environ.get("LOG_PATH_FWD", ".")
    if world_size == "1":
        save_file = os.path.join(log_path_fwd, f"{date}-single.pt")
    else:
        path = os.path.join(log_path_fwd, date)
        os.makedirs(path, exist_ok=True)
        save_file = os.path.join(path, f"rank{rank}.pt")
    torch.save(LOGGED_TENSORS, save_file)
    print(f"[Rank {rank}] 📝 Logged fwd tensors saved to {save_file}")


def save_logged_grads(rank, world_size, date):
    global LOGGED_GRADS
    log_path_bwd = os.environ.get("LOG_PATH_BWD", ".")
    if world_size == "1":
        save_file = os.path.join(log_path_bwd, f"{date}-grad.pt")
    else:
        path = os.path.join(log_path_bwd, date)
        os.makedirs(path, exist_ok=True)
        save_file = os.path.join(path, f"rank{rank}-grad.pt")
    torch.save(LOGGED_GRADS, save_file)
    print(f"[Rank {rank}] ⛰️ Logged bwd tensors saved to {save_file}")


def save_tensors():
    rank = os.environ.get("RANK", "0")
    world_size = os.environ.get("WORLD_SIZE", "1")
    global LOGGED_TENSORS, LOGGED_GRADS, LOGGED_G_OPTIM
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    if not LOGGED_TENSORS:
        if rank == 0:
            print(f"⛔️ No fwd tensors logged.")
    else:
        save_logged_tensors(rank, world_size, date)
    if not LOGGED_GRADS:
        if rank == 0:
            print(f"⛔️ No bwd tensors logged.")
    else:
        save_logged_grads(rank, world_size, date)
    


if os.environ.get("LOG_PATH_FWD", "") != "":
    logging_tensor = _logging_tensor
if os.environ.get("LOG_PATH_BWD", "") != "":
    logging_grad = _logging_grad
if os.environ.get("LOG_PATH_FWD", "") or os.environ.get("LOG_PATH_BWD", ""):
    atexit.register(save_tensors)
