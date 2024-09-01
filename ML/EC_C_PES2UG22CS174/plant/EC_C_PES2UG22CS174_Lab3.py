import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    target_column = tensor[:, -1]
    unique_elements, counts = torch.unique(target_column, return_counts=True)
    probabilities = counts.float() / counts.sum()
    entropy = -torch.sum(probabilities * torch.log2(probabilities))
    return entropy.item()

def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    attribute_column = tensor[:, attribute]
    target_column = tensor[:, -1]
    unique_elements, counts = torch.unique(attribute_column, return_counts=True)
    avg_info = 0.0
    for value, count in zip(unique_elements, counts):
        subset = tensor[attribute_column == value]
        subset_entropy = get_entropy_of_dataset(subset)
        avg_info += (count.item() / tensor.size(0)) * subset_entropy
    return avg_info

def get_information_gain(tensor: torch.Tensor, attribute: int):
    entropy_before_split = get_entropy_of_dataset(tensor)
    avg_info_of_attribute = get_avg_info_of_attribute(tensor, attribute)
    information_gain = entropy_before_split - avg_info_of_attribute
    return information_gain

def get_selected_attribute(tensor: torch.Tensor):
    num_attributes = tensor.size(1) - 1
    information_gains = {}
    for attribute in range(num_attributes):
        ig = get_information_gain(tensor, attribute)
        information_gains[attribute] = ig
    selected_attribute = max(information_gains, key=information_gains.get)
    return information_gains, selected_attribute
