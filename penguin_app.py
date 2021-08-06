import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

@st.cache()
def prediction(model,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex):
  species = model.predict([[island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex]])
  if species[0]==0 :
    return 'Adelie'
  if species[0]==1 :
    return 'Chinstrap' 
  if species[0]==2 :
    return 'Gentoo' 

st.sidebar.title('PENGUIN SPECIES CLASSIFICATION APP')
blm = st.sidebar.slider('bill length mm',float(df['bill_length_mm'].min()),float(df['bill_length_mm'].max()))
bdm = st.sidebar.slider('bill depth mm',float(df['bill_depth_mm'].min()),float(df['bill_depth_mm'].max()))
flm = st.sidebar.slider('flipper length mm',float(df['flipper_length_mm'].min()),float(df['flipper_length_mm'].max()))
bmg = st.sidebar.slider('body mass (in g)',float(df['body_mass_g'].min()),float(df['body_mass_g'].max()))

sex = st.sidebar.selectbox('sex',('Male','Female'))
sex_mapped = pd.Series(sex).map({'Male':0,'Female':1})

island = st.sidebar.selectbox('island',('Biscoe','Dream','Torgersen'))
island_mapped = pd.Series(island).map({'Biscoe':0,'Dream':1,'Torgersen':2})


classifier = st.sidebar.selectbox('Classifier',('Random Forest Classifier','Support Vector Machine','Logistic Regression'))
if st.sidebar.button('Predict'):

  if classifier=='Random Forest Classifier':
    predicted = prediction(rf_clf,island_mapped,blm,bdm,flm,bmg,sex_mapped)
    score = rf_clf.score(X_train,y_train)

  if classifier=='Logistic Regression':
    predicted = prediction(log_reg,island_mapped,blm,bdm,flm,bmg,sex_mapped)
    score = log_reg.score(X_train,y_train)

  elif classifier=='Support Vector Machine':
    predicted = prediction(svc_model,island_mapped,blm,bdm,flm,bmg,sex_mapped)
    score = svc_model.score(X_train,y_train)
  st.write('The predicted species is ' , predicted)
  st.write('The accuracy of the model is ', round(score,2))