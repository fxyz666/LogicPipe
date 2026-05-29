"""
SOT decoding pipeline.
"""
import torch
import torch.distributed as dist
from core.core.decoding_communication import CommunicationHandler
import time
from  tasks.medusa_llama.outline_decoding_controller  import get_controller   # [MODIFIED]


class DecodingPipeline():
    def __init__(self,  stage_model, config, args):
        self.config = config
        self.args = args
        self.stage = config.stage
        self.next_rank = config.next_rank
        self.pre_rank = config.pre_rank
        self.total_stage = config.total_stage
        self.stage_model = stage_model
        self.comm_handler = CommunicationHandler(config)
        
    def tree_decoding_send(self, tensor, point_id):
        assert self.stage != self.total_stage -1
        self.comm_handler.send(tensor, tag = self.comm_handler.tensor_tag["tree_decoding"], point_id=point_id)
        
    def tree_decoding_recv(self):
        assert self.stage != 0
        tensor,point_id= self.comm_handler.recv(tag = self.comm_handler.tensor_tag["tree_decoding"])
        return tensor,point_id
    
    def tree_candidates_send(self, tensor, point_id):
        assert self.comm_handler.if_last_rank
        self.comm_handler.send(tensor, tag = self.comm_handler.tensor_tag["tree_candidates"], point_id=point_id)
        
    def tree_candidates_recv(self):
        assert self.comm_handler.if_first_rank
        tensor,point_id= self.comm_handler.recv(tag = self.comm_handler.tensor_tag["tree_candidates"])
        return tensor,point_id
    
    def new_token_send(self, tensor, point_id):
        assert self.comm_handler.if_last_rank
        self.comm_handler.send(tensor,tag = self.comm_handler.tensor_tag["new_token"], point_id=point_id)
    def new_token_recv(self):
        assert not self.comm_handler.if_last_rank
        tensor,point_id = self.comm_handler.recv(self.comm_handler.tensor_tag["new_token"])
        return tensor,point_id

    def _broadcast_has_request(self, controller) -> bool:
        run_flag = torch.zeros(1, dtype=torch.int64)
        if self.config.is_last_stage and controller.has_pending_requests():
            run_flag[0] = 1
        dist.broadcast(run_flag, src=self.total_stage - 1)
        return bool(run_flag.item())

    def jupiter_decoding_step(self) -> bool:
        # Use point_id to select point_kv_cache in forward computation or modify point_kv_cache based on selected token.
        controller = get_controller()
        if controller.point_num == 0:
            return False
        if controller.all_points_finished():
            return False
        if not self._broadcast_has_request(controller):
            return False
        extra_kwargs = {
                'is_point': True,
                'point_id': 0,
                }
        print("=========================================\n")
        # step 1: get request (medusa_logits and logits) from request queue and then generate_candidates
        new_token=0 # no use
        if self.config.is_last_stage:
            request = controller.get_request()
            candidates, tree_candidates = self.stage_model.generate_candidates(
                request["medusa_logits"], 
                request["logits"], 
            )
            input_ids = controller.get_input_ids( request["point_id"] )
            self.tree_candidates_send(tree_candidates, request["point_id"] )
        if self.config.is_first_stage:
            tree_candidates, point_id = self.tree_candidates_recv()
            input_ids = controller.get_input_ids( point_id)
        # Step 2: tree decoding
        if self.config.is_first_stage:
            if  self.config.is_last_stage:
                raise NotImplementedError("Single-machine inference not supported yet.")
            extra_kwargs["point_id"]=point_id
            hidden_states = self.stage_model.tree_decoding(
                tree_candidates = tree_candidates,
                tree_candidates_embeds = None,
                input_ids = input_ids,
                **extra_kwargs  # Pass extra parameters.
            )
            self.tree_decoding_send(hidden_states,point_id)
        else:
            hidden_states, point_id = self.tree_decoding_recv()
            input_ids = controller.get_input_ids(point_id)
            extra_kwargs["point_id"]=point_id
            if not self.config.is_last_stage:
                hidden_states = self.stage_model.tree_decoding(
                tree_candidates = None,
                tree_candidates_embeds = hidden_states,
                input_ids = input_ids,
                **extra_kwargs  # Pass extra parameters.
                )
                self.tree_decoding_send(hidden_states,point_id)
            else:
                medusa_logits, logits  = self.stage_model.tree_decoding(
                tree_candidates = None,
                tree_candidates_embeds = hidden_states,
                input_ids = input_ids,
                **extra_kwargs  # Pass extra parameters.
                )
        # Step 3: Select the best candidate as the sampling result and update inputs_ids.
        if self.config.is_last_stage:
            best_candidate, accept_length = self.stage_model.evaluate_posterior(logits,
                        candidates)
            extra_kwargs["point_id"]=point_id
            input_ids, logits, medusa_logits, new_token , select_indices= self.stage_model.update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            logits,
            medusa_logits,
            new_token,
            **extra_kwargs
            )  
            new_input_ids = input_ids[:,  -select_indices.shape[0]:]
            # update input_ids request
            controller.update_input_ids(input_ids,point_id)
            input_len = controller.get_input_len(point_id)
            if self.stage_model.tokenizer.eos_token_id in input_ids[0, input_len:]:
                print("Point: {} Finish".format(point_id),flush=True)
                controller.finalize_point(point_id)
            else:
                controller.add_request(medusa_logits,logits,point_id)
            # TODO: Determine if finished.
        # Step 4: In Step 3, self.stage_model.update_inference_inputs updated the kvcache of the last stage. Now synchronize the decoding results and update the kv cache and inputs_ids of other stages.
        if  self.config.is_last_stage:
            new_token_len =  torch.tensor(select_indices.shape)
            select_indices_and_new_inputs_ids = torch.cat((new_token_len.unsqueeze(0), select_indices.unsqueeze(0).cpu(), new_input_ids.cpu()), dim=1)
            self.new_token_send(select_indices_and_new_inputs_ids,point_id)
        else:
            select_indices_and_new_inputs_ids, point_id = self.new_token_recv()
            new_token_len = select_indices_and_new_inputs_ids[0,0].item()
            select_indices = select_indices_and_new_inputs_ids[:,1:new_token_len+1].view(-1)
            new_input_ids = select_indices_and_new_inputs_ids[:, new_token_len+1:2*new_token_len+1]
            extra_kwargs["point_id"]=point_id
            self.stage_model.update_kv_cache(input_ids,select_indices,**extra_kwargs)
            input_ids = torch.cat([input_ids, new_input_ids], dim=-1    ) # Must be executed after update_kv_cache.
            # update input_ids
            controller.update_input_ids(input_ids,point_id)
            input_len = controller.get_input_len(point_id)
            if self.stage_model.tokenizer.eos_token_id in input_ids[0, input_len:]:
                print("Point: {} Finish".format(point_id),flush=True)
                controller.finalize_point(point_id)
        return True

    def jupiter_decoding_pipeline(self, max_steps=None):
        controller = get_controller()
        max_decode_steps = self.config.max_steps * max(1, controller.point_num)
        if max_steps is not None:
            max_decode_steps = min(max_decode_steps, max_steps)
        executed_steps = 0
        for idx in range(max_decode_steps):
            if controller.all_points_finished():
                break
            print("step {}".format(idx),flush=True)
            if not self.jupiter_decoding_step():
                break
            executed_steps += 1
        return executed_steps

