import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.linear_model import LogisticRegression


def separate_features_and_labels(echo_tracks):
    #you don't need track_id,genre_top in features
    features = echo_tracks.drop(['genre_top','track_id'],axis=1)
    labels = echo_tracks['genre_top']

    return features,labels

def split_dataset_and_standarize(features,labels):
    #split the dataset
    #obtain train_features, test_features, train_labels, test_labels 
    train_features, test_features, train_labels, test_labels= 
    scaled_train_features = 
    scaled_test_features = 

    return scaled_train_features,scaled_test_features,train_labels,test_labels

def get_explained_variance(scaled_train_features):
    #use PCA on scaled train features and obtain exp_variance
    exp_variance = 
    return exp_variance

def get_number_of_features(exp_variance):
    cum_exp_variance = np.cumsum(exp_variance)

    # Plot the cumulative explained variance and draw a dashed line at 0.85.
    fig, ax = plt.subplots()
    ax.plot(cum_exp_variance)
    ax.axhline(y=0.85, linestyle='--')
    ax.set_xlim(0, len(exp_variance))  # Extend x-axis to include all components
    ax.set_ylim(0, 2.0)  # Extend y-axis from 0 to 1 (assuming explained variance is in percentage)

    # Add labels and title
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Cumulative Explained Variance vs Number of Components')
    #if explained variance =1 then it represents all components are included , there is no point of dimensionality reduction
    # Show the plot
    plt.show()
    
    # find the required number of components 
    req_no_of_comp = 

    return req_no_of_comp

def perform_pca(scaled_train_features,scaled_test_features,no_of_features):
    # Perform PCA with the chosen number of components and project data onto components


    # Fit and transform the scaled training features using pca
    

    # Fit and transform the scaled test features using pca
    

    return train_pca , test_pca

def train_and_pred_decision_tree(train_pca,test_pca,train_labels):
    #train using DecisionTreeClassifier
    return pred_labels_tree

def train_and_pred_logistic(train_pca,test_pca,train_labels):
    #train using logistic regression model
    

    return pred_labels_logit

def get_accuracy(test_labels,pred_labels_tree):
    class_rep_tree = classification_report(test_labels,pred_labels_tree)
    #print("Decision Tree: \n", class_rep_tree)
    return class_rep_tree


def balancing_of_dataset(echo_tracks):
    #Balance the dataset
    #Subset a balanced proportion of data points
    hop_only = echo_tracks.loc[echo_tracks['genre_top'] == 'Hip-Hop']
    rock_only = echo_tracks.loc[echo_tracks['genre_top'] == 'Rock']

    # subset only the rock songs, and take a sample the same size as there are hip-hop songs
    rock_only = 

    # concatenate the dataframes hop_only and rock_only
    rock_hop_bal = pd.concat([rock_only, hop_only])

    return rock_hop_bal
