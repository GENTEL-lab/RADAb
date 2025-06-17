import torch

def char_to_index(char):
    # 创建一个映射，将字符映射到一个唯一的索引
    char_to_index = {c: i for i, c in enumerate('abcdefghijklmnopqrstuvwxyz0123456789')}
    return char_to_index[char]

def pdbid_to_tensor(pdb_id):
    # 将每个字符转换为索引，然后转换为张量
    indices = [char_to_index(char) for char in pdb_id]
    return torch.tensor(indices, dtype=torch.long)

def index_to_char(index):
    # 创建一个映射，将索引映射回字符
    index_to_char = {i: c for i, c in enumerate('abcdefghijklmnopqrstuvwxyz0123456789')}
    return index_to_char[index]

def tensor_to_pdbid(tensor):
    return ''.join([index_to_char(index.item()) for index in tensor])
