# -*- coding: utf-8 -*-
# @Time : 20-6-11 上午10:19
# @Author : zhuying
# @Company : Minivision
# @File : iccv_flops_calculation.py
# @Software : PyCharm
import torch
import torch.nn as nn

multiply_adds = 1


def count_convNd(m, x, y):
    cin = m.in_channels

    kernel_ops = m.weight.size()[2:].numel()
    bias_ops = 0 if m.bias is not None else -1
    ops_per_element = kernel_ops
    output_elements = y.nelement()

    # cout x oW x oH
    total_ops = (2 * cin * ops_per_element + bias_ops) * output_elements // m.groups
    m.total_ops = torch.Tensor([int(total_ops)])


def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    bias_ops = 1 if m.bias is not None else 0
    total_ops = (total_mul + total_add + bias_ops) * num_elements

    m.total_ops = torch.Tensor([int(total_ops)])


register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.Linear: count_linear,
}


def profile(model, input_size, custom_ops={}, device="cpu"):
    handler_collection = []

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        m_type = type(m)
        fn = None

        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
        else:
            print("Not implemented for ", m)

        if fn is not None:
            print("Register FLOP counter for module %s" % str(m))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    original_device = model.parameters().__next__().device
    training = model.training

    model.eval().to(device)
    model.apply(add_hooks)
    x = torch.zeros(input_size).to(device)
    with torch.no_grad():
        model(x)

    total_ops = 0
    total_params = 0
    n_layer = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        ops_layer = m.total_ops
        params_layer = m.total_params
        total_ops += m.total_ops
        total_params += m.total_params
        n_layer += 1
        print('./.......................................................................................................')
        print('layer:', n_layer)
        print(m)
        print('params_layer:', int(params_layer), 'ops_layer:', int(ops_layer), 'total_params:', int(total_params), 'total_ops:', int(total_ops))


    total_ops = total_ops.item()
    total_params = total_params.item()

    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()
    print(model)


    return total_ops, total_params
