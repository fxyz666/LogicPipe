from .prefilling_pipeline import PrefillingPipeline
from .decoding_pipeline import DecodingPipeline
import time
import torch
import copy
import torch.distributed as dist
from .core.tag_manager   import Tag
from typing import Callable, Optional

def jupiter_prefilling(input_ids,model,config,args ):
    # Prefilling with sequence slicing.
    if config.device == "cuda" and input_ids.device != "cuda":
        input_ids = input_ids.cuda()
    print("input_ids shape:",input_ids.shape)
    ######################################################################
    # Pipelined Prefilling Stage
    start = time.time()
    runtime = PrefillingPipeline(model, config, args)
    if config.is_last_stage:    # Last stage generates medusa logits for decoding stage.
        medusa_logits, logits = runtime.pipeline_with_sequence_slicing( )
    else:
        if config.is_first_stage:
            runtime.pipeline_with_sequence_slicing(input_ids) # or use runtime.pipeline_forward
        else:
            runtime.pipeline_with_sequence_slicing()
    print("prefilling time:", time.time() - start)
    # runtime.comm_handler.stop_helper_threads()
    if config.is_last_stage:
        return medusa_logits, logits
    else:
        return None, None
def jupiter_prefilling_no_finish(input_ids,model,config,args   ):
    # Prefilling with sequence slicing.
    # Used for shared prefix filling.
    # Final hidden_states do not go through lm_heads or medusa_head.
    # No need to set_mask_for_medusa_decoding.
    if config.device == "cuda" and input_ids.device != "cuda":
        input_ids = input_ids.cuda()
    print("input_ids shape:",input_ids.shape)
    ######################################################################
    runtime = PrefillingPipeline(model, config, args)
    if config.is_last_stage:
        runtime.pipeline_with_sequence_slicing_no_finish( )
    else:
        if config.is_first_stage:
            runtime.pipeline_with_sequence_slicing_no_finish(input_ids)
        else:
            runtime.pipeline_with_sequence_slicing_no_finish()

def point_prefilling(points,model,config,args, point_ids=None ):
    '''
    Prefilling points for every request.
    e.g.
    points[0] =  `1. [/INST]1. Exercise`
    points[1] =  '2. [/INST]2. Mindfulness`
    Then get medusa_logits and logits for decoding phase (only for last stage).
    '''
    tokenizer = model.get_tokenizer()
    runtime = PrefillingPipeline(model, config, args)
    points_input_ids = [tokenizer.encode(point, return_tensors="pt") for point in  points]
    points_input_ids = [point[:,2:] for point in points_input_ids]
    if config.device == "cuda":
        points_input_ids = [p.cuda() for p in points_input_ids ]
    if config.is_last_stage:
        medusa_logits_list,logits_list = runtime.points_saturation(points_input_ids, point_ids=point_ids)
    else:
        runtime.points_saturation(points_input_ids, point_ids=point_ids)
    # runtime.comm_handler.stop_helper_threads()
    if config.is_last_stage:
        return medusa_logits_list,logits_list
    else:
        return None ,None
class NormalDecodingSession:
    def __init__(self, prompt, model, config, medusa_logits=None, logits=None, input_ids=None):
        self.prompt = prompt
        self.model = model
        self.config = config
        tag_manager = Tag()
        self.tensor_tag = {
            "tree_decoding": tag_manager.get_next_tag(),
            "tree_candidates": tag_manager.get_next_tag(),
            "new_token": tag_manager.get_next_tag(),
        }
        self.tensor_shape = {
            "tree_decoding": (1, 64, config.hidden_size),
            "tree_candidates": (1, 64),
            "new_token": (1, 1 + 2 * config.medusa_num_heads),
        }
        self.tensor_type = {
            "tree_decoding": config.torch_dtype,
            "tree_candidates": torch.int64,
            "new_token": torch.int64,
        }
        tokenizer = model.get_tokenizer()
        if input_ids is None:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
        if config.device == "cuda":
            input_ids = input_ids.cuda()
        self.input_ids = input_ids
        self.input_len = input_ids.shape[1]
        self.medusa_logits = medusa_logits
        self.logits = logits
        self.new_token = 0
        self.text = ""
        self.step_idx = 0
        self.finished = False

    def step(self) -> str:
        if self.finished:
            return self.text

        model = self.model
        config = self.config
        input_ids = self.input_ids
        text = self.text

        if config.is_last_stage:
            candidates, tree_candidates = model.generate_candidates(
                self.medusa_logits,
                self.logits,
            )
        if config.is_first_stage:
            recv_tensor = torch.zeros(
                self.tensor_shape["tree_candidates"],
                dtype=self.tensor_type["tree_candidates"],
            )
            dist.recv(
                tensor=recv_tensor,
                src=config.total_stage - 1,
                tag=self.tensor_tag["tree_candidates"],
            )
            tree_candidates = recv_tensor.to("cuda") if config.device == "cuda" else recv_tensor

        if config.is_last_stage:
            send_tensor = tree_candidates.cpu() if config.device == "cuda" else tree_candidates
            dist.send(tensor=send_tensor, dst=0, tag=self.tensor_tag["tree_candidates"])

        if config.is_first_stage:
            if config.is_last_stage:
                raise NotImplementedError("Single-machine inference not supported yet.")
            hidden_states = model.tree_decoding(
                tree_candidates=tree_candidates,
                tree_candidates_embeds=None,
                input_ids=input_ids,
            )
            dist.send(
                tensor=hidden_states.cpu(),
                dst=config.next_rank,
                tag=self.tensor_tag["tree_decoding"],
            )
        else:
            recv_tensor = torch.zeros(
                self.tensor_shape["tree_decoding"],
                dtype=self.tensor_type["tree_decoding"],
            )
            dist.recv(tensor=recv_tensor, src=config.pre_rank, tag=self.tensor_tag["tree_decoding"])
            hidden_states = recv_tensor.to("cuda") if config.device == "cuda" else recv_tensor

            if not config.is_last_stage:
                hidden_states = model.tree_decoding(
                    tree_candidates=None,
                    tree_candidates_embeds=hidden_states,
                    input_ids=input_ids,
                )
                dist.send(
                    tensor=hidden_states.cpu(),
                    dst=config.next_rank,
                    tag=self.tensor_tag["tree_decoding"],
                )
            else:
                self.medusa_logits, self.logits = model.tree_decoding(
                    tree_candidates=None,
                    tree_candidates_embeds=hidden_states,
                    input_ids=input_ids,
                )

        if config.is_last_stage:
            best_candidate, accept_length = model.evaluate_posterior(self.logits, candidates)
            input_ids, self.logits, self.medusa_logits, self.new_token, select_indices = model.update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                self.logits,
                self.medusa_logits,
                self.new_token,
            )
            new_input_ids = input_ids[:, -select_indices.shape[0]:]
            text = model.tokenizer.decode(
                input_ids[0, self.input_len:],
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
        if config.is_last_stage:
            new_token_len = torch.tensor(select_indices.shape)
            select_indices_and_new_inputs_ids = torch.cat(
                (
                    new_token_len.unsqueeze(0),
                    select_indices.unsqueeze(0).cpu(),
                    new_input_ids.cpu(),
                ),
                dim=1,
            )
            dist.broadcast(select_indices_and_new_inputs_ids, src=config.total_stage - 1)
        else:
            recv_tensor = torch.zeros(
                self.tensor_shape["new_token"],
                dtype=self.tensor_type["new_token"],
            )
            dist.broadcast(recv_tensor, src=config.total_stage - 1)
            select_indices_and_new_inputs_ids = recv_tensor.cuda() if config.device == "cuda" else recv_tensor
            new_token_len = select_indices_and_new_inputs_ids[0, 0].item()
            select_indices = select_indices_and_new_inputs_ids[:, 1:new_token_len + 1].view(-1)
            new_input_ids = select_indices_and_new_inputs_ids[:, new_token_len + 1:2 * new_token_len + 1]
            if config.device == "cuda":
                select_indices = select_indices.cuda()
            model.update_kv_cache(input_ids, select_indices)
            if config.device == "cuda":
                new_input_ids = new_input_ids.cuda()
            input_ids = torch.cat([input_ids, new_input_ids], dim=-1)
            text = model.tokenizer.decode(
                input_ids[0, self.input_len:],
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

        self.input_ids = input_ids
        self.text = text
        self.step_idx += 1
        self.finished = (
            model.tokenizer.eos_token_id in input_ids[0, self.input_len:]
            or self.step_idx >= config.max_steps
        )
        return self.text

    def run(self, on_text_update: Optional[Callable[[str], None]] = None) -> str:
        while not self.finished:
            text = self.step()
            if on_text_update is not None:
                on_text_update(text)
        return self.text


def normal_decoding(prompt,model,config,medusa_logits=None,logits=None,input_ids=None,on_text_update=None):
    session = NormalDecodingSession(
        prompt=prompt,
        model=model,
        config=config,
        medusa_logits=medusa_logits,
        logits=logits,
        input_ids=input_ids,
    )
    return session.run(on_text_update=on_text_update)
def outline_based_decoding( model,config,args, max_steps=None ):
    model.set_mask_for_medusa_decoding()
    runtime = DecodingPipeline(model, config, args)
    return runtime.jupiter_decoding_pipeline(max_steps=max_steps)
