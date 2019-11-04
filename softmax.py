import numpy as np
import matplotlib.pyplot as plt

def mySoftmax1(x):
    sum_exp = np.sum(np.exp(x))
    return list(map(lambda i: np.exp(i)/sum_exp, x))

def mySoftmax2(x):
    nx = list(map(lambda i: np.exp(i - np.max(x)), x))
    return nx / np.sum(nx)

lst = [1.0, 2.0, 2.0]
y1, y2 = mySoftmax1(lst), mySoftmax2(lst)

print(y1)
print(y2)

plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.pie(y1, labels=y1, shadow=True,startangle=90)
plt.subplot(122)
plt.pie(y2, labels=y2, shadow=True,startangle=90)
plt.show()