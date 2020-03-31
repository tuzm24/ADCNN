import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np
from help_func.__init__ import ExecFileName
import os
module_name_dic = {}

def summary(model, input_size, batch_size=-1, device="cuda"):
    def register_hook(module):

        def hook(module, input):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            if class_name=='Conv2d':
                class_name +=str(module.kernel_size)
            if class_name not in module_name_dic:
                module_name_dic[class_name] = 1
            module_idx = len(summary)
            m_key = "%s%s-%i" % (depth[0]*'    ',class_name, module_name_dic[class_name])
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
            depth[0] +=1

        def outhook(moudle, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            if class_name=='Conv2d':
                class_name +=str(module.kernel_size)
            if class_name not in module_name_dic:
                module_name_dic[class_name] = 1
            module_idx = len(summary)

            m_key = "%s%s-%i" % ((depth[0]-1)*'    ',class_name, module_name_dic[class_name])
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size
            module_name_dic[class_name] +=1
            depth[0] -=1

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            # and not (module == model)
        ):
            hooks.append((module.register_forward_pre_hook(hook), module.register_forward_hook(outhook)))


    def applyfirst(module, fn):
        fn(module)
        for mod in module.children():
            applyfirst(mod, fn)
        return module


    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    depth = [0]
    hooks = []

    # register hook
    applyfirst(model, register_hook)
    # model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h, h2 in hooks:
        h.remove()
        h2.remove()

    print("--------------------------------------------------------------------------------------------------------------")
    line_new = "{:<40}  {:>25}  {:>25} {:>15}".format("Layer (type)", "Input Shape", "Output Shape", "Param #")
    print(line_new)
    print("==============================================================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:<40}  {:>25}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]['input_shape']),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("==============================================================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("--------------------------------------------------------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("--------------------------------------------------------------------------------------------------------------")
    # return summary


def summary_to_file(model, input_size, path, batch_size=-1, device="cuda",):
    def register_hook(module):

        def hook(module, input):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            if class_name=='Conv2d':
                class_name +=str(module.kernel_size)
            if class_name not in module_name_dic:
                module_name_dic[class_name] = 1
            module_idx = len(summary)
            m_key = "%s%s-%i" % (depth[0]*'    ',class_name, module_name_dic[class_name])
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
            depth[0] +=1

        def outhook(moudle, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            if class_name=='Conv2d':
                class_name +=str(module.kernel_size)
            if class_name not in module_name_dic:
                module_name_dic[class_name] = 1
            module_idx = len(summary)

            m_key = "%s%s-%i" % ((depth[0]-1)*'    ',class_name, module_name_dic[class_name])
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size
            module_name_dic[class_name] +=1
            depth[0] -=1

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            # and not (module == model)
        ):
            hooks.append((module.register_forward_pre_hook(hook), module.register_forward_hook(outhook)))


    def applyfirst(module, fn):
        fn(module)
        for mod in module.children():
            applyfirst(mod, fn)
        return module


    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # f.write(type(x[0]))

    # create properties
    summary = OrderedDict()
    depth = [0]
    hooks = []

    # register hook
    applyfirst(model, register_hook)
    # model.apply(register_hook)

    # make a forward pass
    # f.write(x.shape)
    model(*x)

    # remove these hooks
    for h, h2 in hooks:
        h.remove()
        h2.remove()
    filename = path
    f = open(filename, 'w')
    f.write("--------------------------------------------------------------------------------------------------------------")
    f.write('\n')
    line_new = "{:<40}  {:>25}  {:>25} {:>15}".format("Layer (type)", "Input Shape", "Output Shape", "Param #")
    f.write(line_new)
    f.write('\n')
    f.write("==============================================================================================================")
    f.write('\n')
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:<40}  {:>25}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]['input_shape']),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        f.write(line_new)
        f.write('\n')

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    f.write("==============================================================================================================\n")
    f.write("Total params: {0:,}\n".format(total_params))
    f.write("Trainable params: {0:,}\n".format(trainable_params))
    f.write("Non-trainable params: {0:,}\n".format(total_params - trainable_params))
    f.write("--------------------------------------------------------------------------------------------------------------\n")
    f.write("Input size (MB): %0.2f\n" % total_input_size)
    f.write("Forward/backward pass size (MB): %0.2f\n" % total_output_size)
    f.write("Params size (MB): %0.2f\n" % total_params_size)
    f.write("Estimated Total Size (MB): %0.2f\n" % total_size)
    f.write("--------------------------------------------------------------------------------------------------------------\n")
    f.close()
    # return summary