import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm

def merge_and_replace (
        indices: list,
        pair: tuple,
        new_index: int,
        **kwargs
        ) -> list:
    new_indices = []
    i = 0 
    while i < len(indices):
        if i+1 < len(indices) and indices[i] == pair[0] and indices[i+1]==pair[1]:
            new_indices.append(new_index)
            i = i+2
        else:
            new_indices.append(indices[i])
            i= i+1

    return new_indices

def max_freq(
        first_list: list,
        second_list: list
) -> tuple:
    char_freq = defaultdict(int)
    for index1, index2 in zip(first_list, second_list):
        char_freq[(index1, index2)] +=1 

    index1, index2 = max(char_freq,key= char_freq.get)
    del char_freq[(index1, index2)]

    return (index1, index2)

    

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int = None,
    special_tokens: list[str] = [],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # count = 0
    indices = []
    char_freq = defaultdict(int)
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    merge_order = {}
    with open(input_path, encoding='utf-8') as f:
        for line in tqdm(f):
            indices.extend(list(map(int,line.encode("utf-8"))))
            # count = count+1
            # if count >50:
            #     break
        
    if vocab_size == None:
        vocab_size = 512


    # print(char_freq)
    
    for i in tqdm(range((vocab_size-256)+1)):
        index1, index2 = max_freq(indices, indices[1:])
        new_index = 256+i
        pair = (index1, index2)
        vocab[new_index]=vocab[index1] + vocab[index2]
        merge_order[(index1, index2)] = new_index
        indices = merge_and_replace(indices, pair, new_index)
        
    for i in range(len(special_tokens)):
        vocab[new_index + i+1] = special_tokens[i]

    

    return (vocab, merge_order)

def encode (
        input_path: str | os.PathLike,
        params: tuple      
) -> list:
    vocab, merge_order = params[0], params[1]
    indices = []
    # count = 0
    print("Reading Encoding File")
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            indices.extend(list(map(int,line.encode("utf-8"))))
            # count = count+1
            # if count >50:
            #     break
    # print("Before encoding")
    # print(indices)
    # print("length of array")
    # print(len(indices))
    print("Starting Encoding Process")
    for keys in tqdm(merge_order.keys()):
        array_index = 0

        while array_index < (len(indices)-1):
            pair = (indices[array_index], indices[array_index+1])
            if keys == pair:                
                indices.insert(array_index,merge_order[pair])
                del indices[array_index + 1:array_index + 3]
                array_index = array_index+2
            else:
                array_index = array_index+1


    # print("after encoding")
    # print(indices)
    # print("length of array after")
    # print(len(indices))

    return indices
    

def decode(
        encoded_list: list,
        params: tuple,
        **kwargs,       
) ->  list:
    vocab, merge_order = params[0], params[1]
    output_string= ""
    print("Starting Decoding process")
    for i in tqdm(range(len(encoded_list))):
        output_string = output_string+((vocab[encoded_list[i]]).decode('utf-8'))

    return output_string
        

      

def main():
    params = train_bpe("C:/Users/anirb/CS336/assignment1-basics_cs336/data/TinyStoriesV2-GPT4-train.txt",300, ['<|endoftext|>'])
    encoded_output = encode("C:/Users/anirb/CS336/assignment1-basics_cs336/data/TinyStoriesV2-GPT4-valid.txt",params)
    test_output = decode(encoded_output,params)
    # with open("C:/Users/anirb/CS336/assignment1-basics_cs336/data/testfile.txt", "w") as f:
    #     f.write(test_output)



if __name__ == "__main__":
    main()