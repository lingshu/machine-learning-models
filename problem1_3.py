import sys
import csv
import numpy as np

data = []
labels = []
with open(sys.argv[1], 'r') as csvfile1:
    orig_data = csv.reader(csvfile1)
    for row in orig_data:
        row_data = [float(i) for i in row[0: 2]]
        row_data.append(1)
        data.append(row_data)
        # labels.append(float(i) for i in row[2])
        labels.append(float(row[2]))

w = np.zeros(len(data[0]))
# w = np.array(map(float, w))
data = np.array(data)
labels = np.array(labels)
error = True
with open(sys.argv[2], 'w') as csvfile2:
    weights = csv.writer(csvfile2)
    while error:
        error = False
        for i in range(len(data)):
            angle = np.dot(data[i], w) * labels[i]
            if angle <= 0:
                error = True
                w_new = labels[i] * data[i]
                w += w_new
        weights.writerow(w)
        # for i, j in zip(data, labels):
        #     x_new = np.dot(i, w)
