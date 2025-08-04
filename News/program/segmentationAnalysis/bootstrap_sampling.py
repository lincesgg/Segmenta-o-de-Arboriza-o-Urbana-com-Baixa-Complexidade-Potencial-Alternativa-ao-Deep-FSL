from ast import List
import random
from typing import Any
from analysisTypes import *

def sample_ds(dataset: t.List[Any], idxs_to_sample_from: t.List[int]) -> t.List[Any]:
    """
    Sample a dataset based on a list of indexes.

    Args:
    - dataset (List[Any]): The dataset to sample from.
    - idxs_to_sample_from (List[int]): The indexes to sample from.

    Returns:
    - List[Any]: The sampled dataset.
    """
    selected_samples = []
    
    for selected_idx in idxs_to_sample_from:
        selected_samples.append(dataset[selected_idx])
        
    return selected_samples


def generate_bootstrap_samples_idxs(dataset_length:int, samples_n:int=1) -> t.List[t.Tuple[t.List[int], t.List[int]]]:
    samples_idxs = []
    
    for i in range(samples_n):
        dataset_idxs = [*range(dataset_length)]

        selected_idxs = random.choices(dataset_idxs, k=dataset_length)
        no_selected_idxs = [*set.difference(set(dataset_idxs), set(selected_idxs))]
        
        samples_idxs.append((selected_idxs, no_selected_idxs))
    
    return samples_idxs

            
def generate_bootstrap_sample_data(dataset:t.List[any], samples_n:int):
    """
    Generate bootstrap samples from a given dataset.

    Args:
    - dataset (List[any]): The dataset to sample from.
    - samples_n (int): The number of samples to generate.

    Returns:
    - List[Tuple[List[any], List[any]]]: A list of tuples, where each tuple contains
      the selected data points and the unselected data points for each sample.
    """
    samples = []
    sample_idxs = generate_bootstrap_samples_idxs(len(dataset), samples_n)
    
    for (selected_idxs, unselected_idxs) in sample_idxs:
        selected_data_points = sample_ds(dataset, selected_idxs)
        unselected_data_points = sample_ds(dataset, unselected_idxs)
        
        samples.append((selected_data_points, unselected_data_points))
    
    return samples


def n_bootstrap_samples(dataset: t.List[any], n:int=10):

    samples = []
    
    for _ in range(n):
        crr_samples = []
        while (len(crr_samples) == 0):
            crr_samples = generate_bootstrap_sample_data(dataset)
            
        samples.append(crr_samples)
        
    return samples
