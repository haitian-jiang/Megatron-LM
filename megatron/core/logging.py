import os
import torch
import atexit
import datetime
from megatron.core import parallel_state


LOGGED_TENSORS = []
local_batch_id = -1


def logging_tensor(name, tensor):
    pass


def logging_tensor_impl(name, tensor):
    if tensor is None:
        return
    
    global LOGGED_TENSORS

    if name == "tokens":
        global local_batch_id
        local_batch_id += 1

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()
    glb_batch_id = local_batch_id * dp_size + dp_rank

    LOGGED_TENSORS.append((f"batch_{glb_batch_id}.{name}", tensor.detach().cpu()))


def save_logged_tensors():
    rank = os.environ.get("RANK", "0")
    world_size = os.environ.get("WORLD_SIZE", "1")
    global LOGGED_TENSORS
    if not LOGGED_TENSORS:
        print(f"[Rank {rank}] ⛔️ No tensors logged. Exiting...")
        return
    # get date of yy-mm-dd-hh-mm
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    if world_size == "1":
        save_path = os.path.join(os.environ.get("LOG_PATH", "."), f"{date}-single.pt")
    else:
        save_path = os.path.join(os.environ.get("LOG_PATH", "."), date, f"rank{rank}.pt")
        # create directory if not exists
        os.makedirs(os.path.join(os.environ.get("LOG_PATH", "."), date), exist_ok=True)
    torch.save(LOGGED_TENSORS, save_path)
    print(f"[Rank {rank}] 📝 Logged tensors saved to {save_path}")


if os.environ.get("LOG_PATH", "") != "":
    logging_tensor = logging_tensor_impl
    atexit.register(save_logged_tensors)
