from sklearn import tree,datasets,linear_model

import numpy as np

iris=datasets.load_iris()



removed=[0,50,100]


new_data=np.delete(iris.data,removed,axis=0)

new_target=np.delete(iris.target,removed)

clf=linear_model.LinearRegression()

clf=clf.fit(new_data,new_target)

predicted=clf.predict(iris.data[removed])
print("removed target are",iris.target[removed])
print("predicted target are",predicted)
