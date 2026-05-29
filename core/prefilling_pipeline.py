import torch

from core.core.schedules import PipelineRuntime

class PrefillingPipeline(PipelineRuntime):
    def __init__(self, stage_model, config, args):
        super().__init__(stage_model, config, args)

    def split_tensor_along_dim(self,tensor, num_splits, dim=1):
        shape = list(tensor.size())
        assert dim < len(shape), "Dimension out of range for the tensor"
        split_size = shape[dim] // num_splits
        remainder = shape[dim] % num_splits
        assert split_size +1 <= self.config.max_sub_sequence_len # +1 is for concatenating sub_seq length information.
        assert remainder +1 <= self.config.max_sub_sequence_len
        # Split tensor.
        splits = []
        start = 0
        for i in range(num_splits):
            length = split_size + 1 if i < remainder else split_size
            splits.append(tensor.narrow(dim, start, length))
            start += length
        return splits

    def pipeline_with_sequence_slicing(self ,input_ids = None):
        if self.config.is_first_stage:
            bs,_ = input_ids.shape
            assert bs == 1
        # Step 0: Initialize prefilling (init kv cache).
        self.stage_model.prefilling_init()
        # Step 1: Split the original sequence into multiple sub-sequences.
        if self.config.is_first_stage:
            assert input_ids is not None
            sub_sequences = self.split_tensor_along_dim(input_ids, self.config.num_sub_sequences, dim=1)
        for i in range (self.config.num_sub_sequences):
            if self.config.is_first_stage:
                if self.config.is_last_stage:
                    raise NotImplementedError("Single-machine inference not supported yet.")
                sub_input_ids = sub_sequences[i]
                seq_len = torch.tensor( int (sub_input_ids.shape[1])).reshape(1,1)
                print("seq_len", seq_len)
                self.send_seq_len(seq_len)
                hidden_states = self.stage_model.forward_sub_sequences(input_ids=sub_input_ids, inputs_embeds=None )
                self.send_activation_forward(hidden_states)
            else:
                seq_len =  int (self.receive_seq_len().item())
                print("seq_len", seq_len)
                hidden_states = self.receive_activation_forward()
                hidden_states = hidden_states[:,:seq_len,:]
                hidden_states = self.stage_model.forward_sub_sequences(input_ids=None, inputs_embeds=hidden_states )
                if not self.config.is_last_stage:   # Not the first or last stage.
                    seq_len = torch.tensor( int (hidden_states.shape[1])).reshape(1,1)
                    self.send_seq_len(seq_len)
                    self.send_activation_forward(hidden_states)

        if self.config.is_last_stage:
            # hidden_states = torch.cat(sub_hidden_states, dim=1)
            print("hidden_states", hidden_states.shape)
            # Step 2: Get medusa_logits and logits.
            medusa_logits,logits = self.stage_model.prefilling_finish(hidden_states)
            print("finish prefilling")
            return medusa_logits, logits
        else:
            self.stage_model.prefilling_finish( )
            print("finish prefilling")

    def pipeline_with_sequence_slicing_no_finish(self ,input_ids = None):
        if self.config.is_first_stage:
            bs,_ = input_ids.shape
            assert bs == 1
        # Step 0: Initialize prefilling (init kv cache).
        self.stage_model.prefilling_init()
        # Step 1: Split the original sequence into multiple sub-sequences.
        if self.config.is_first_stage:
            assert input_ids is not None
            sub_sequences = self.split_tensor_along_dim(input_ids, self.config.num_sub_sequences, dim=1)
        for i in range (self.config.num_sub_sequences):
            if self.config.is_first_stage:
                sub_input_ids = sub_sequences[i]
                if self.config.is_last_stage:
                    raise NotImplementedError("Single-machine inference not supported yet.")
                seq_len = torch.tensor( int (sub_input_ids.shape[1])).reshape(1,1)
                self.send_seq_len(seq_len)
                hidden_states = self.stage_model.forward_sub_sequences(input_ids=sub_input_ids, inputs_embeds=None )
                self.send_activation_forward(hidden_states)
            else:
                seq_len =  int (self.receive_seq_len().item())
                hidden_states = self.receive_activation_forward()
                hidden_states = hidden_states[:,:seq_len,:]
                hidden_states = self.stage_model.forward_sub_sequences(input_ids=None, inputs_embeds=hidden_states )
                if not self.config.is_last_stage:   # Not the first or last stage.
                    seq_len = torch.tensor( int (hidden_states.shape[1])).reshape(1,1)
                    self.send_seq_len(seq_len)
                    self.send_activation_forward(hidden_states)


    def points_saturation(self,points_input_ids, point_ids=None):
        medusa_logits_list = []
        logits_list = []
        extra_kwargs = {
            'is_point': True,
            'point_id':0,
                    }
        if point_ids is None:
            point_ids = list(range(len(points_input_ids)))
        for point_id, input_ids in zip(point_ids, points_input_ids):
            if self.config.is_first_stage:
                if  self.config.is_last_stage:
                    raise NotImplementedError("Single-machine inference not supported yet.")
                # Set is_point to True.
                extra_kwargs["point_id"]=point_id
                hidden_states = self.stage_model.forward_sub_sequences(input_ids=input_ids, inputs_embeds=None, **extra_kwargs)
                seq_len = torch.tensor( int (hidden_states.shape[1])).reshape(1,1)
                self.send_seq_len(seq_len)
                self.send_activation_forward(hidden_states)
            else:
                seq_len =  int (self.receive_seq_len().item())
                hidden_states = self.receive_activation_forward()
                hidden_states = hidden_states[:,:seq_len,:]
                extra_kwargs["point_id"]=point_id
                hidden_states = self.stage_model.forward_sub_sequences(input_ids=None, inputs_embeds=hidden_states, **extra_kwargs)

                if not self.config.is_last_stage:   # Not the first or last stage.
                    seq_len = torch.tensor( int (hidden_states.shape[1])).reshape(1,1)
                    self.send_seq_len(seq_len)
                    self.send_activation_forward(hidden_states)
                else: # Last stage.
                    medusa_logits = []
                    with torch.inference_mode():
                        logits =  self.stage_model.lm_head(hidden_states)
                        for i in range(self.config.medusa_num_heads):
                            medusa_logits.append(self.stage_model.medusa_head[i](hidden_states))
                        medusa_logits_list.append( torch.stack(medusa_logits, dim=0))
                        logits_list.append(logits)

        if self.config.is_last_stage:
            return  medusa_logits_list,logits_list
        else:
            return None
