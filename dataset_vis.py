from os import name
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('dataset.csv', names=['label', 'data'])

labels = data['label']
label_dict = {}

for i in labels:
    if i not in label_dict:
        label_dict[i] = 1
    else:
        label_dict[i] += 1

fig1, ax1 = plt.subplots()
ax1.pie(label_dict.values(), labels=label_dict.keys(), autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')

plt.savefig('pie_chart.pdf')