from dataclasses import dataclass
from typing import List

import torch


class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.
        current_length (int): Current length of the data being stored.
    """

    def __init__(self, data, current_length):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int): Initial length of the data.
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.

        Args:
            indices (torch.Tensor): Indices of the data tensor to be copied.
            prev_length (int): Previous length before adding new data.
            dim (int, optional): Dimension along which copying should be performed. Default is 2.
        """
        tgt = self.data.index_select(dim, indices)
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation should be done. Default is 2.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
        dst = self.data.narrow(dim, self.current_length, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        return torch.narrow(self.data, 2, 0, self.current_length)
    # [modified]
    def get_states(self):
        return torch.narrow(self.data, 2, 0, self.current_length)


@dataclass
class ContextualCacheUnit:
    """Reusable point-level KV snapshot for dependency context."""

    key_states_by_layer: List[torch.Tensor]
    value_states_by_layer: List[torch.Tensor]
    token_count: int


def reset_kv_cache(past_key_values, current_length_data) -> None:
    current_length_data.zero_()


def restore_contextual_cache(
    past_key_values,
    current_length_data,
    cache_unit: ContextualCacheUnit,
) -> None:
    reset_kv_cache(past_key_values, current_length_data)
    append_contextual_cache_unit(past_key_values, cache_unit)

def initialize_past_key_values(model):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to store and reuse
    past key and value states during autoregressive decoding, which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states need to be initialized.

    Returns:
        tuple:
            - past_key_values (list): A list of KVCache objects for each layer in the model.
            - past_key_values_data (torch.Tensor): The tensor that will store all keys and values.
            - current_length_data (torch.Tensor): A tensor tracking the current length of keys/values in the cache.
    """
    # Extracting configuration from the model
    config = model.config
    # Initializing the batch size to 1, this can be modified if different batch sizes are required
    batch_size = 1
    # Initializing a tensor to store past keys and values for all layers
    if hasattr(config, "num_pp_hidden_layers") and config.num_pp_hidden_layers != None:
        # [modified]
        config.num_hidden_layers = config.num_pp_hidden_layers
    print("num_hidden_layers", config.num_hidden_layers)
    if hasattr(config, "max_kv_cache_length"):
        max_length = config.max_kv_cache_length
    else:
        max_length = config.max_position_embeddings
    past_key_values_data = torch.zeros(
            config.num_hidden_layers * 2, 
            batch_size,
            config.num_key_value_heads,
            max_length, # [modified]
            config.hidden_size // config.num_attention_heads,
            device=model.device,
            dtype=model.dtype,
        )
 
        
    print("kv cache mem",past_key_values_data.element_size() * past_key_values_data.numel()/(1024*1024))
    # Initialize tensor to store the current length of the cached data for all layers.
    # [IMPORTANT] It needs to be kept on CPU for quick access and updates.
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    # Creating a KVCache for each pair of key and value in all layers
    past_key_values = [] * config.num_hidden_layers
    for i in range(config.num_hidden_layers):
        past_key_values.append(
            [
                KVCache(past_key_values_data[i * 2 + j], current_length_data[i * 2 + j])
                for j in range(2)
            ]
        )
    return past_key_values, past_key_values_data, current_length_data

def get_shared_kv_and_point_kv( shared_past_key_value,point_past_key_value):
    share_key_states = shared_past_key_value[0].get_states()
    share_value_states  = shared_past_key_value[1].get_states()
    
    
    point_key_states = point_past_key_value[0].get_states()
    point_value_states  = point_past_key_value[1].get_states()
    
    # 按照最后一个维度（seqlen 维度）进行拼接
    combined_key_states = torch.cat((share_key_states, point_key_states), dim=2)
    combined_value_states = torch.cat((share_value_states, point_value_states), dim=2)
    return combined_key_states, combined_value_states


def snapshot_contextual_cache(point_past_key_values) -> ContextualCacheUnit:
    key_states_by_layer: List[torch.Tensor] = []
    value_states_by_layer: List[torch.Tensor] = []
    token_count = 0
    for layer_key_value in point_past_key_values:
        key_states = layer_key_value[0].get_states().clone()
        value_states = layer_key_value[1].get_states().clone()
        token_count = max(token_count, int(key_states.shape[2]))
        key_states_by_layer.append(key_states)
        value_states_by_layer.append(value_states)
    return ContextualCacheUnit(
        key_states_by_layer=key_states_by_layer,
        value_states_by_layer=value_states_by_layer,
        token_count=token_count,
    )


def clear_contextual_cache(past_key_values) -> None:
    for layer_key_value in past_key_values:
        layer_key_value[0].current_length.fill_(0)
        layer_key_value[1].current_length.fill_(0)


def append_contextual_cache_unit(point_past_key_values, cache_unit: ContextualCacheUnit) -> None:
    for layer_idx, layer_key_value in enumerate(point_past_key_values):
        layer_key_value[0].cat(cache_unit.key_states_by_layer[layer_idx], dim=2)
        layer_key_value[1].cat(cache_unit.value_states_by_layer[layer_idx], dim=2)


def load_contextual_cache(past_key_values, cache_unit: ContextualCacheUnit) -> None:
    clear_contextual_cache(past_key_values)
    append_contextual_cache_unit(past_key_values, cache_unit)

