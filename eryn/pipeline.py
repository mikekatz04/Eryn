from abc import ABC

import numpy as np


class InfoManager:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.update_info(**kwargs)

    def update_info(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class PipelineModule(ABC):
    def __init__(self, name=None):
        self.name = name

    @classmethod
    def initialize_module(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def update_module(self, info_manager, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def run_module(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def update_information(self, *args, **kwargs):
        raise NotImplementedError


class PipelineGuide:
    def __init__(self, info_manager, module_list):
        self.module_list = module_list
        self.info_manager = info_manager

    def run(self, progress=False, verbose=False):
        for i, module in enumerate(self.module_list):
            if verbose:
                print_str = "starting module {}".format(i)
                if module.name is not None:
                    print_str += ": {}".format(module.name)
                print(print_str)

            module.update_module(self.info_manager)
            module.run_module(progress=progress)

            if verbose:
                print_str = "finished module {}".format(i)
                if module.name is not None:
                    print_str += ": {}".format(module.name)
                print(print_str)

            # TODO: add output manager
