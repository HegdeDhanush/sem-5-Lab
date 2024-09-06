import sys
import importlib
import argparse
import torch 
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--ID", required=True)

args = parser.parse_args()
subname = args.ID

try:
    mymodule = importlib.import_module(subname)
except Exception as e:
    print("Rename your written program as CAMPUS_SECTION_SRN_Lab1.py and run python Test.py --ID CAMPUS_SECTION_SRN_Lab1")
    sys.exit()

separate_features_and_labels = mymodule.separate_features_and_labels
split_dataset_and_standarize = mymodule.split_dataset_and_standarize
get_explained_variance = mymodule.get_explained_variance
get_number_of_features= mymodule.get_number_of_features
# fit_scaled_train_test_features=mymodule.fit_scaled_train_test_features
perform_pca = mymodule.perform_pca
train_and_pred_decision_tree=mymodule.train_and_pred_decision_tree
get_accuracy=mymodule.get_accuracy
balance_dataset = mymodule.balancing_of_dataset
train_and_pred_logistic=mymodule.train_and_pred_logistic
def run_tests(dataset):
    
    #separate features and labels
    features,labels = separate_features_and_labels(dataset)
    #scale the features
    scaled_train_features , scaled_test_features , train_labels , test_labels = split_dataset_and_standarize(features,labels)
    print("features and labels are obtained --before balancing of the dataset")
    #performing dimensionality reduction
    exp_variance = get_explained_variance(scaled_train_features)
    #the number of features required to yield optimum results
    no_of_features = get_number_of_features(exp_variance)
    print("No of feature are",no_of_features)
    #perform pca for dimensionality reduction
    train_pca , test_pca = perform_pca(scaled_train_features,scaled_test_features,no_of_features)
    #pred_labels_tree
    pred_labels_tree = train_and_pred_decision_tree(train_pca,test_pca,train_labels)
    #get results
    class_rep_tree = get_accuracy(test_labels,pred_labels_tree)
    print("DECISION TREE RESULTS:")
    print(class_rep_tree)
    print("LOGISTIC REGRESSION RESULTS:")
    pred_labels_log = train_and_pred_logistic(train_pca,test_pca,train_labels)
    class_rep_log=get_accuracy(test_labels,pred_labels_log)
    print(class_rep_log)


    

    ######################################################
    ## AFTER BALANCING OF THE DATASET
    #balance the dataset
    print("############BALANCING OF THE DATASET##################")
    rock_hop_balance = balance_dataset(dataset)
    #obtain features and labels
    features_new,labels_new = separate_features_and_labels(rock_hop_balance)
    #obtain scaled features
    scaled_train_features_new , scaled_test_features_new , train_labels_new , test_labels_new = split_dataset_and_standarize(features_new,labels_new)
    #
    print("features and labels are obtained --before balancing of the dataset")
    exp_variance_new = get_explained_variance(scaled_train_features_new)
    #the number of features required to yield optimum results
    no_of_features_new = get_number_of_features(exp_variance_new)
    print("No of feature are",no_of_features)
    #perform pca for dimensionality reduction
    train_pca_new , test_pca_new = perform_pca(scaled_train_features_new,scaled_test_features_new,no_of_features_new)
    #pred_labels_tree
    pred_labels_tree_new = train_and_pred_decision_tree(train_pca_new,test_pca_new,train_labels_new)
    #get results
    class_rep_tree_new = get_accuracy(test_labels_new,pred_labels_tree_new)
    print("DECISION TREE RESULTS:")
    print(class_rep_tree_new)
    print("LOGISTIC REGRESSION RESULTS:")
    pred_labels_log_new = train_and_pred_logistic(train_pca_new,test_pca_new,train_labels_new)
    class_rep_log_new=get_accuracy(test_labels_new,pred_labels_log_new)
    print(class_rep_log_new)


if __name__=="__main__":
    echo_tracks = pd.read_csv('echo_tracks.csv')

    run_tests(echo_tracks)