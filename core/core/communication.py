"""
Each device needs to continuously receive data from the previous and next devices,
and add them to the forward and backward task queues.
"""

import threading
import torch
import torch.distributed as dist
from . import threadsafe_queue,tag_manager

class CommunicationHandler():
    """Handles communication between stages."""
    def __init__(self, config):
        self.rank = config.stage
        self.world_size = config.total_stage
        self.next_rank = config.next_rank
        self.pre_rank = config.pre_rank
        self.if_first_rank = config.is_first_stage
        self.if_last_rank = config.is_last_stage
        self.tag_manager = tag_manager.Tag()
        self.tensor_tag = {"forward": self.tag_manager.get_next_tag(),
                            "seq_len": self.tag_manager.get_next_tag()}
        self.tensor_type = {"forward":config.torch_dtype,
                            "seq_len":torch.int64}
        # For variable sequence length, this is set to max_sub_sequence_len.
        self.tensor_shape = {"forward": (1, config.max_sub_sequence_len, config.hidden_size),
                            "seq_len": (1,1)}
        self.device = config.device
        self.setup_queue()
        # Stop event to signal threads to stop.
        self.stop_event = threading.Event()
        # List to keep track of helper threads.
        self.helper_threads = []
        self.start_helper_threads()
    def setup_queue(self):
        """
        Set up queues for communication between main compute thread
        and helper communication threads. One queue per tensor
        in forward/backward direction.
        """
        if not self.if_first_rank:
            self.forward_receive_queues = threadsafe_queue.Queue()
            self.seq_len_receive_queues = threadsafe_queue.Queue()
        if not self.if_last_rank:
            self.forward_send_queues = threadsafe_queue.Queue()
            self.seq_len_send_queues = threadsafe_queue.Queue()

    def start_helper_threads(self):
        # Prefill uses a short-lived communication handler per prefill pass.
        # Keeping background recv threads alive after the pass can block later
        # collectives on the same process group, so sends/recvs are synchronous.
        return

    def start_helper_thread(self, func, args):
        helper_thread = threading.Thread(target=func, args=args,daemon=True)
        helper_thread.start()
        self.helper_threads.append(helper_thread)  # Track the thread.

    def stop_helper_threads(self):
        # Signal all helper threads to stop.
        # self.stop_event.set()
        pass
        
    def send(self, tensor, tag):
        if tag == self.tensor_tag["forward"]:
            _send(tensor, self.next_rank, tag)
        elif  tag == self.tensor_tag["seq_len"]:
            _send(tensor, self.next_rank, tag)
        else:
            raise NotImplementedError

    def recv(self, tag):
        if tag == self.tensor_tag["forward"]:
            tensor = _recv(
                self.tensor_shape["forward"],
                self.pre_rank,
                tag,
                self.tensor_type["forward"],
            )
            tensor = tensor.requires_grad_()
        elif tag == self.tensor_tag["seq_len"]:
            tensor = _recv(
                self.tensor_shape["seq_len"],
                self.pre_rank,
                tag,
                self.tensor_type["seq_len"],
            )
        else:
            raise NotImplementedError
        if self.device == "cuda":
            tensor = tensor.cuda()
        return tensor


def recv_helper_thread(recv_queue, tensor_shape, src_rank, tag, dtype, stop_event):
    """
    Thread responsible for receiving tensors.

    Arguments:
        - recv_queue: Receive queue.
        - tensor_shape: Shape of the tensor.
        - src_rank: Rank of the sender.
        - tag: Tag of the tensor.
        - dtype: Data type of the tensor.
        - stop_event: Stop event.
    """
    while not stop_event.is_set():  # Check if stop signal is set.
        tensor = _recv(tensor_shape, src_rank, tag,dtype)
        recv_queue.add(tensor)


def send_helper_thread(send_queue, dst_rank, tag, stop_event):
    """
    Thread responsible for sending tensors.

    Arguments:
        - send_queue: Queue of tensors waiting to be sent.
        - dst_rank: Rank of the destination.
        - tag: Tag of the tensor.
        - stop_event: Stop event.
    """
    while not stop_event.is_set():  # Check if stop signal is set.
        # Queue blocks when send_queue is empty.
        tensor = send_queue.remove()
        _send(tensor, dst_rank, tag,)
# TODO: Define backend (gloo), only supports CPU.
def _send(tensor, dst_rank, tag ):
    if tensor.device != torch.device("cpu"):
        tensor = tensor.cpu()
    dist.send(tensor=tensor, dst=dst_rank, tag=tag)



def _recv(tensor_shape, src_rank, tag,dtype):
    tensor = torch.zeros(tensor_shape, dtype=dtype) 
    dist.recv(tensor, src=src_rank, tag=tag)
    return tensor
