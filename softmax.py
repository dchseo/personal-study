import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

lst = [1.0, 2.0, 2.0]

# softmax standard formulation
def mySoftmax1(x):
    sum_exp = np.sum(np.exp(x))
    return list(map(lambda i: np.exp(i)/sum_exp, x))

# softmax formulation with subtracting max-value
def mySoftmax2(x):
    nx = list(map(lambda i: np.exp(i - np.max(x)), x))
    return nx / np.sum(nx)

# tensorflow softmax function (tensorflow ver 2.0)
v = tf.constant(lst, name='vector')
r = tf.nn.softmax(v)
print(r)

y1, y2 = mySoftmax1(lst), mySoftmax2(lst)

print("ver1: ", y1)
print("ver2: ", y2)

plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.pie(y1, labels=y1, shadow=True,startangle=90)
plt.subplot(122)
plt.pie(y2, labels=y2, shadow=True,startangle=90)
plt.show()