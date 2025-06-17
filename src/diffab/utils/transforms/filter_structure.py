import torch

from ..protein import constants
from ._base import register_transform

class MissingChainException(Exception):
    pass

@register_transform('filter_structure')
class FilterStructure(object):

    def __init__(self, must_have_heavy=True, must_have_light=True, must_have_antigen=True):
        super().__init__()
        self.must_have_heavy = must_have_heavy
        self.must_have_light = must_have_light
        self.must_have_antigen = must_have_antigen
        
    

    def __call__(self, structure):
        if self.must_have_heavy and structure['heavy'] is None:
            raise MissingChainException('heavy chain missing')
        if self.must_have_light and structure['light'] is None:
            raise MissingChainException('light chain missing')
        if self.must_have_antigen and structure['antigen'] is None:
            raise MissingChainException('antigen missing')
        
        return structure

