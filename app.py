import streamlit as st
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("mid_term_model_practice.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('Dataset.csv')
X = dataset.iloc[:,0:14].values

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NaN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.
imputer = imputer.fit(X[:,1:5])
#Replacing missing data with the calculated mean value
X[:,1:5]= imputer.transform(X[:,1:5])

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NaN, strategy= 'constant', fill_value='Male', verbose=1, copy=True)
#Fitting imputer object to the independent variables x.
imputer = imputer.fit(X[:, 5:6])
#Replacing missing data with the constant value
X[:, 5:6]= imputer.transform(X[:,5:6])

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NaN, strategy= 'constant', fill_value='France', verbose=1, copy=True)
#Fitting imputer object to the independent variables x.
imputer = imputer.fit(X[:, 6:7])
#Replacing missing data with the constant value
X[:, 6:7]= imputer.transform(X[:,6:7])


# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NaN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.
imputer = imputer.fit(X[:, 7:14])
#Replacing missing data with the calculated mean value
X[:, 7:14]= imputer.transform(X[:, 7:14])

# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 5] = labelencoder_X.fit_transform(X[:,5])

# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 6] = labelencoder_X.fit_transform(X[:,6])



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

def predict_note_authentication(age,cp,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal):
  output= model.predict(sc.transform([[age,cp,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal]]))
  print("Heart Disease Category is",output)
  if output==[0]:
    prediction="Heart Disease Category is 0"


  if output==[1]:
    prediction="Heart Disease Category is 1 "


  if output==[2]:
    prediction="Heart Disease Category is 2"


  if output==[3]:
    prediction="Heart Disease Category is 3"

  if output==[4]:
    prediction="Heart Disease Category is 4 "

  print(prediction)
  return prediction
def main():

    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center>
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center>
   <center><p style="font-size:25px;color:white;margin-top:10px;">Deep Learning  Lab Experiment Deployment</p></center>
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Item Purchase Prediction")
    age = st.number_input('Insert a Age',18,60)
    cp = st.number_input('Insert a cp',1,5)
    trestbps = st.number_input('Insert a trestbps',100,200)
    chol = st.number_input('Insert a chol',100,500)
    fbs = st.number_input('Insert a fbs',0,1)
    Gender = st.number_input('Insert Gender')
    Geography= st.number_input('Insert Geography')
    restecg= st.number_input('Insert a restecg',0,250)
    thalach = st.number_input('Insert a thalach',0,200)
    exang = st.number_input('Insert a exang',0,10)
    oldpeak = st.number_input('Insert a oldpeak',0,5)
    slope = st.number_input('Insert a slope',0,3)
    ca= st.number_input('Insert a ca',0,7)
    thal = st.number_input('Insert a thal',0,7)

    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(age,cp,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Yash Koolwal")
      st.subheader("C-Section,PIET")

if __name__=='__main__':
  main()