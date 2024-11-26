import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# read csv
data = np.genfromtxt('Inference_test/6_pred.csv', delimiter=',', skip_header=0)

ground_truth_binary = data[:, :1]
prediction_binary = data[:, 1:]
ground_truth_binary_flat = ground_truth_binary.flatten()

# cal
conf_matrix = confusion_matrix(ground_truth_binary_flat, prediction_binary)


plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', cbar=True, xticklabels=['H0', 'F1','F2', 'F3'], yticklabels=['H0', 'F1','F2', 'F3'])
plt.xlabel('Predicted Label', fontsize=18)
plt.ylabel('True Label', fontsize=18)


# save
plt.savefig('confusion_matrix.png', dpi=900, bbox_inches='tight')
plt.show()






