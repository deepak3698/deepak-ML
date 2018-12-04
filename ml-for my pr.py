import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
###data for the value ei,e2,e3,e4,e5
features=[[32,3,121,2113,111],[43,12313,1313,12313,55]]
np.append(features,[17,12,2131,13131,1313])
labels=[[2],[5]]
f1=[[242424,2424,242,24241,3131]]
print(features)
print(labels)
print(f1)


model = linear_model.LinearRegression()

model.fit(features,labels)

predicted = model.predict(f1)

print(predicted)

if(predicted<2):
    print("yes we are using the same things")
else:
    print("I am sorry to  say that  i can not")

#plt.scatter(features, labels,color="blue")
#plt.plot(features, labels)

#plt.show()


