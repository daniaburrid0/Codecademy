'''
In this project, you will be using K-means clustering (the algorithm behind this magic)
and scikit-learn to cluster images of handwritten digits.
'''
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
# print(digits)

# print(digits.DESCR)
# print(digits.data)

# print(digits.target)

# plt.gray()
# plt.matshow(digits.images[100])
# plt.show()

# print(digits.target[100])

# fig = plt.figure(figsize=(6, 6))
 
# # Adjust the subplots 
 
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
 
# # For each of the 64 images
 
# for i in range(64):
 
#     # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
 
#     ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
 
#     # Display an image at the i-th position
 
#     ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
 
#     # Label the image with the target value
 
#     ax.text(0, 7, str(digits.target[i]))
 
# plt.show()


k = 10
model = KMeans(n_clusters=k, random_state=42)

model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))
 
fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(k):
    ax = fig.add_subplot(2, 5, 1 + i)
    ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()

new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.15,5.49,3.51,0.00,0.00,0.00,0.00,0.00,2.90,7.62,4.57,0.00,0.00,0.00,0.00,0.00,5.72,7.62,4.57,0.00,0.00,0.00,0.00,0.08,7.17,7.62,4.57,0.00,0.00,0.00,0.00,1.22,7.62,7.63,4.57,0.00,0.00,0.00,0.00,0.23,2.82,6.25,5.26,0.00,0.00,0.00,0.00,0.00,0.00,4.42,4.88,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.92,3.58,6.03,7.63,3.74,0.00,0.00,0.00,4.27,7.63,5.80,6.56,5.95,0.00,0.00,0.00,0.38,1.07,0.00,5.26,6.10,0.00,0.00,0.00,0.00,0.00,3.05,7.62,3.66,0.00,0.00,0.00,0.00,3.20,7.62,7.47,3.35,1.68,0.00,0.00,0.00,1.91,6.03,7.25,7.62,4.73,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.31,2.36,5.41,7.55,7.55,1.83,0.00,0.00,5.03,7.62,6.79,3.97,7.24,3.81,0.00,0.00,1.37,2.36,0.15,3.96,7.62,3.89,0.00,0.00,0.00,0.00,0.46,7.32,7.62,7.40,0.31,0.00,0.00,0.00,0.00,1.83,5.03,7.62,0.76,0.00,0.00,0.00,5.19,7.62,7.62,5.19,0.08,0.00,0.00,0.00,2.44,2.97,0.38,0.00,0.00,0.00],
[0.00,0.15,2.75,0.15,0.53,0.84,0.00,0.00,0.00,1.52,7.62,1.53,5.19,6.02,0.00,0.00,0.00,1.83,7.62,1.52,6.10,6.10,0.00,0.00,0.00,2.29,7.62,4.96,7.17,6.10,0.00,0.00,0.00,1.37,6.10,6.10,7.32,5.19,0.00,0.00,0.00,0.00,0.00,0.00,6.10,4.57,0.00,0.00,0.00,0.00,0.00,0.00,6.10,4.57,0.00,0.00,0.00,0.00,0.00,0.00,3.81,2.59,0.00,0.00]
])

new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
    if new_labels[i] == 0:
        print(0, end='')
    elif new_labels[i] == 1:
        print(9, end='')
    elif new_labels[i] == 2:
        print(2, end='')
    elif new_labels[i] == 3:
        print(1, end='')
    elif new_labels[i] == 4:
        print(6, end='')
    elif new_labels[i] == 5:
        print(8, end='')
    elif new_labels[i] == 6:
        print(4, end='')
    elif new_labels[i] == 7:
        print(5, end='')
    elif new_labels[i] == 8:
        print(7, end='')
    elif new_labels[i] == 9:
        print(3, end='')