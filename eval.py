import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt


# Function to visualize a confusion matrix
def plot_confusion_matrix(confusion_matrix, class_names):
    # Normalize the confusion matrix
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    # Fix nans
    confusion_matrix = np.nan_to_num(confusion_matrix)

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Use white text if squares are dark; otherwise black
    threshold = confusion_matrix.max() * 0.85
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        color = "white" if confusion_matrix[i, j] > threshold else "black"
        plt.text(j, i, "{:0.4f}".format(confusion_matrix[i, j]), horizontalalignment="center", color=color)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def get_accuracy_per_class(model, dataloader, device, pred_fn=lambda x: torch.argmax(x, dim=1)):
    """Return tuple of the accuracy per class for a given model and dataloader (test loader), and confusion matrix.
    The model must be in evaluation mode. The dataloader must have a .dataset attribute with a .num_classes attribute"""
    # Create a confusion matrix
    confusion_matrix = np.zeros((dataloader.dataset.num_classes, dataloader.dataset.num_classes))
    
    # Put the model in evaluation mode
    model.eval()
    
    # Iterate over the data in the dataloader
    with torch.no_grad():
        for batch in dataloader:
            # Get the inputs and labels
            inputs = batch[0]
            labels = batch[1]
            
            # Move them to the GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get the predicted class with the highest score
            predicted = pred_fn(outputs)
            
            # Update the confusion matrix
            for i in range(len(predicted)):
                confusion_matrix[int(labels[i]), int(predicted[i])] += 1
    
    # Compute the accuracy per class
    accuracy_per_class = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

    # Fix nans
    accuracy_per_class[np.isnan(accuracy_per_class)] = 0
    confusion_matrix[np.isnan(confusion_matrix)] = 0
    
    return accuracy_per_class, confusion_matrix
