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
    # We don't need track_id and genre_top in features
    features = echo_tracks.drop(['genre_top', 'track_id'], axis=1)
    labels = echo_tracks['genre_top']

    return features, labels

def split_dataset_and_standarize(features, labels):
    # Split the dataset into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(train_features)
    scaled_test_features = scaler.transform(test_features)

    return scaled_train_features, scaled_test_features, train_labels, test_labels

def get_explained_variance(scaled_train_features):
    # Use PCA on scaled train features and obtain explained variance
    pca = PCA()
    pca.fit(scaled_train_features)
    exp_variance = pca.explained_variance_ratio_
    
    return exp_variance

def get_number_of_features(exp_variance):
    cum_exp_variance = np.cumsum(exp_variance)

    # Plot the cumulative explained variance and draw a dashed line at 0.85.
    fig, ax = plt.subplots()
    ax.plot(cum_exp_variance)
    ax.axhline(y=0.85, linestyle='--')
    ax.set_xlim(0, len(exp_variance))
    ax.set_ylim(0, 1.0)

    # Add labels and title
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Cumulative Explained Variance vs Number of Components')
    
    # Show the plot
    plt.show()
    
    # Find the required number of components to explain at least 85% of the variance
    req_no_of_comp = np.argmax(cum_exp_variance >= 0.85) + 1

    return req_no_of_comp

def perform_pca(scaled_train_features, scaled_test_features, no_of_features):
    # Perform PCA with the chosen number of components and project data onto components
    pca = PCA(n_components=no_of_features)

    # Fit and transform the scaled training features using PCA
    train_pca = pca.fit_transform(scaled_train_features)

    # Transform the scaled test features using PCA
    test_pca = pca.transform(scaled_test_features)

    return train_pca, test_pca

def train_and_pred_decision_tree(train_pca, test_pca, train_labels):
    # Train using DecisionTreeClassifier
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(train_pca, train_labels)
    pred_labels_tree = tree_model.predict(test_pca)
    
    return pred_labels_tree

def train_and_pred_logistic(train_pca, test_pca, train_labels):
    # Train using logistic regression model
    logit_model = LogisticRegression(random_state=42)
    logit_model.fit(train_pca, train_labels)
    pred_labels_logit = logit_model.predict(test_pca)
    
    return pred_labels_logit

def get_accuracy(test_labels, pred_labels_tree):
    class_rep_tree = classification_report(test_labels, pred_labels_tree)
    print("Decision Tree: \n", class_rep_tree)
    return class_rep_tree

def balancing_of_dataset(echo_tracks):
    # Balance the dataset by subsampling the majority class
    hop_only = echo_tracks.loc[echo_tracks['genre_top'] == 'Hip-Hop']
    rock_only = echo_tracks.loc[echo_tracks['genre_top'] == 'Rock']

    # Subset only the rock songs, and take a sample the same size as there are hip-hop songs
    rock_only = rock_only.sample(n=len(hop_only), random_state=42)

    # Concatenate the dataframes hop_only and rock_only
    rock_hop_bal = pd.concat([rock_only, hop_only])

    return rock_hop_bal