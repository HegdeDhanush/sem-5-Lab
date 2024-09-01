import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    """Calculate the entropy of the entire dataset"""
    
    target_column = tensor[:, -1]  # Extract the target column (last column)
    unique_elements, counts = torch.unique(target_column, return_counts=True)
    probabilities = counts.float() / counts.sum()  # Calculate the probability of each class
    
    entropy = -torch.sum(probabilities * torch.log2(probabilities))  # Calculate entropy
    
    return entropy.item()  # Return the entropy as a float


# input:tensor,attribute number 
# output:int/float
def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """Return avg_info of the attribute provided as parameter"""
    
    attribute_column = tensor[:, attribute]  # Extract the attribute column
    target_column = tensor[:, -1]  # Extract the target column
    
    unique_elements, counts = torch.unique(attribute_column, return_counts=True)
    avg_info = 0.0
    
    for value, count in zip(unique_elements, counts):
        subset = tensor[attribute_column == value]  # Subset where attribute equals the unique value
        subset_entropy = get_entropy_of_dataset(subset)
        avg_info += (count.item() / tensor.size(0)) * subset_entropy  # Weighted sum of entropies
        
    return avg_info  # Return the average information as a float


# input:tensor,attribute number
# output:int/float
def get_information_gain(tensor: torch.Tensor, attribute: int):
    """Return Information Gain of the attribute provided as parameter"""
    
    entropy_before_split = get_entropy_of_dataset(tensor)  # Entropy of the dataset
    avg_info_of_attribute = get_avg_info_of_attribute(tensor, attribute)  # Avg info of the attribute
    
    information_gain = entropy_before_split - avg_info_of_attribute  # Information gain
    
    return information_gain  
# input: tensor
# output: ({dict},int)
def get_selected_attribute(tensor: torch.Tensor):
    """
    Return a tuple with the first element as a dictionary which has IG of all columns
    and the second element as an integer representing attribute number of selected attribute
    """
    
    num_attributes = tensor.size(1) - 1  # Exclude the target column
    information_gains = {}
    
    for attribute in range(num_attributes):
        ig = get_information_gain(tensor, attribute)
        information_gains[attribute] = ig  # Store IG for each attribute
    
    selected_attribute = max(information_gains, key=information_gains.get)  # Attribute with max IG
    
    return information_gains, selected_attribute  # Return dictionary of IGs and selected attribute