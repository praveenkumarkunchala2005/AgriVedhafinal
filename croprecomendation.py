import pandas as pd

df = pd.read_csv('Crop_recommendation.csv')

df.drop(['ph','humidity','temperature'],axis=1,inplace=True)

df.drop(['rainfall'],axis=1,inplace=True)

x = df.drop(['label'], axis=1)
y = df['label']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,test_size=0.2)

import pickle
from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier()
model2.fit(x_train, y_train)

# Save the trained model using pickle
try:
    with open('crop_recommendation_model.pkl', 'wb') as model_file:
        pickle.dump(model2, model_file)
    print("Model saved successfully.")
except Exception as e:
    print("Error saving the model:", e)
