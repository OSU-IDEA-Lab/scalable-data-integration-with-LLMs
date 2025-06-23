import argparse
import inspect


def get_class_dict(module):
    class_dict = {}
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if "Abstract" not in name and name != "__class__":
            class_dict[name] = cls
    return class_dict

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')