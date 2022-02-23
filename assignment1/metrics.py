import numpy as np
def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification
    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp = np.sum(prediction[ground_truth == True] == True)
    tn = np.sum(prediction[ground_truth == False] == False)
    fp = np.sum(prediction[ground_truth == False] == True)
    fn = np.sum(prediction[ground_truth == True] == False)
    
    eps = 1e-10 # for numerical stability
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps) 

    return precision, recall, f1, accuracy



def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return (prediction == ground_truth).sum() / len(prediction)
