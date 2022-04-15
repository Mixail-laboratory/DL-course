
def binary_classification_metrics(prediction, ground_truth):
    tp = np.sum(prediction[ground_truth == True] == True)
    tn = np.sum(prediction[ground_truth == False] == False)
    fp = np.sum(prediction[ground_truth == False] == True)
    fn = np.sum(prediction[ground_truth == True] == False)
    
    eps = 1e-10 # for numerical stability
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps) 

    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    return (prediction == ground_truth).sum() / len(prediction)