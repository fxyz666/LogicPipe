import torch
from core.core.communication import CommunicationHandler


class PipelineRuntime():
    def __init__(self, stage_model, config, args):
        self.config = config
        self.args = args
        self.stage = config.stage
        self.next_rank = config.next_rank
        self.pre_rank = config.pre_rank
        self.total_stage = config.total_stage
        self.stage_model = stage_model
        self.comm_handler = CommunicationHandler(config)
    
    def send_activation_forward(self, tensor):
        """Send activations in the forward pass.
        """
        # Last stage returns directly.
        if self.stage == self.total_stage-1:
            return
        if tensor.shape[1] > self.config.max_sub_sequence_len:
            raise ValueError(
                "forward activation sequence length exceeds max_sub_sequence_len: "
                f"{tensor.shape[1]} > {self.config.max_sub_sequence_len}"
            )
        if tensor.shape[1] < self.config.max_sub_sequence_len:
            padded_tensor = tensor.new_zeros(
                tensor.shape[0],
                self.config.max_sub_sequence_len,
                tensor.shape[2],
            )
            padded_tensor[:, : tensor.shape[1], :] = tensor
            tensor = padded_tensor
        self.comm_handler.send(tensor, tag = self.comm_handler.tensor_tag["forward"])

    def receive_activation_forward(self, input_sample = None):
        if self.stage == 0: # Get input from input_sample.
            if input_sample is not None:
                if self.config.device == "cuda":
                    tensor = input_sample.cuda()
                else:
                    tensor = input_sample.cpu()
                return tensor
            else:
                raise Exception("Missing input.")
        else: # Receive tensor from previous machine.
            tensor = self.comm_handler.recv(tag = self.comm_handler.tensor_tag["forward"])
        return tensor
        
    def send_seq_len(self, tensor):
        if self.stage == self.total_stage-1:
            return 
        self.comm_handler.send(tensor, tag = self.comm_handler.tensor_tag["seq_len"])
    def receive_seq_len(self):
        tensor = self.comm_handler.recv(tag = self.comm_handler.tensor_tag["seq_len"])
        return tensor
