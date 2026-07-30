import torch
ops = torch.ops._my_softmax_function_2af76fc

def add_op_namespace_prefix(op_name: str):
    """
    Prefix op by namespace.
    """
    return f"_my_softmax_function_2af76fc::{op_name}"