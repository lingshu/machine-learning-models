import sys
import csv
import numpy as np
from decimal import Decimal

def data_process(data):
    data = np.array(data)
    dataProcessed = (data - np.mean(data, axis = 0)) / np.std(data, axis = 0)
    return dataProcessed

def gradient_descent(dataProcessed, labels, alphas, parameters):
    for alpha in alphas:
        risk = 0.0
        betas = np.zeros(len(dataProcessed[0]))
        for i in range(100):
            risk = np.sum(np.square(np.dot(dataProcessed, betas) - labels)) / (2 * len(labels))
            betas -= alpha / len(labels) * np.sum(np.transpose(np.multiply((np.dot(dataProcessed, betas) - labels), np.transpose(dataProcessed))), 0)
        # parameters.writerow((alpha, 100, Decimal(betas[0]).quantize(Decimal('0.00')),Decimal(betas[1]).quantize(Decimal('0.00')), Decimal(betas[2]).quantize(Decimal('0.00'))))
        parameters.writerow((alpha, 100, Decimal(betas[0]), Decimal(betas[1]), Decimal(betas[2])))
        print(risk)

def my_gradient_descent(dataProcessed, labels, alpha, iter, parameters):
    risk = 0.0
    betas = np.zeros(len(dataProcessed[0]))
    for i in range(iter):
        risk = np.sum(np.square(np.dot(dataProcessed, betas) - labels)) / (2 * len(labels))
        betas -= alpha / len(labels) * np.sum(np.transpose(np.multiply((np.dot(dataProcessed, betas) - labels), np.transpose(dataProcessed))), 0)
    parameters.writerow((alpha, iter, Decimal(betas[0]), Decimal(betas[1]), Decimal(betas[2])))
    print(risk)

if __name__ == '__main__':
    age = []
    weight = []
    height = []
    data = []
    labels = []
    with open(sys.argv[1], 'r') as csvfile1:
        orig_data = csv.reader(csvfile1)
        for row in orig_data:
            row_data = [float(i) for i in row[: -1]]
            data.append(row_data)
            labels.append(float(row[-1]))

    with open(sys.argv[2], 'w') as csvfile2:
        parameters = csv.writer(csvfile2)
        dataProcessed = data_process(data)
        dataProcessed = np.c_[np.ones(len(labels)), dataProcessed]
        alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        gradient_descent(dataProcessed, labels, alphas, parameters)
        my_gradient_descent(dataProcessed, labels, 0.8, 60, parameters)



