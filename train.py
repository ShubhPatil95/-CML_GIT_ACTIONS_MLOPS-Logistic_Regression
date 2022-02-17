import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

x=[1,2,3,4,5,6,7,8,9,10]
y=["A","B","A","A","A","B","B","A","B","A"]

df=pd.DataFrame({"X":x,"Y":y})

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values

model=LogisticRegression(random_state=0)
model.fit(x,y)

Y_Pred = model.predict(x)

from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y, Y_Pred)

import seaborn as sns
sns_plot=sns.heatmap(cf_matrix, annot=True)
fig = sns_plot.get_figure()
fig.savefig("confusion_matrix.png")

score=model.score(x, y)
with open("Outputs.txt","w") as file:
    file.write(f"Accuracy is {score}")
