# Importing Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn import preprocessing 
from scipy import stats
import numpy as np
import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
data=''

# Setting up the header 
st.title("-- Regression --")
st.subheader("Complete Regression Model Lifecycle")


Choose_file = st.selectbox("Select the number of files to upload type", ("Single_file", "Two_file",))

if Choose_file == "Single_file":
    filename = st.file_uploader("Upload file", type=("csv", "xlsx"))
    data = pd.read_csv(filename, na_values=['?', '/', '#', ''])
elif Choose_file == 'Two_file':

      # Upload the first dataset
    df1 = st.file_uploader("Upload the first dataset", type=["csv", "xlsx"])

    # Upload the second dataset
    df2 = st.file_uploader("Upload the second dataset", type=["csv", "xlsx"])

    # Merge the two datasets
    if df1 is not None and df2 is not None:
        df1 = pd.read_csv(df1,na_values=['?', '/', '#','']) # Use pd.read_excel(df1) for Excel files
        df2 = pd.read_csv(df2,na_values=['?', '/', '#','']) # Use pd.read_excel(df2) for Excel files
        data = pd.merge(df1, df2, on=['id'])
        st.write(data)
    else:
        st.write("Please upload both datasets.")


def mean_squared_error1(y_true, y_pred):
   
      # Check if the lengths of both arrays are equal
      if len(y_true) != len(y_pred):
          raise ValueError("Length of y_true and y_pred should be the same.")
      
      # Calculate the squared differences between the true and predicted values
      squared_differences = [(y_true[i] - y_pred[i])**2 for i in range(len(y_true))]
      
      # Calculate the mean of the squared differences
      mse1 = sum(squared_differences) / len(squared_differences)
      print(mse1)
      return mse1

from sklearn.metrics import r2_score

def r2(y_true, y_pred):
    # Calculate the mean of the true values
    y_true_mean = sum(y_true) / len(y_true)

    # Calculate the total sum of squares (TSS)
    tss = sum((y_true - y_true_mean) ** 2)

    # Calculate the residual sum of squares (RSS)
    rss = sum((y_true - y_pred) ** 2)

    # Calculate the R-squared value
    r2_score = 1-(rss / tss)

    return r2_score

def remove_outliers(data):
    z_scores = np.abs(stats.zscore(data))
    data_clean = data[(z_scores < 3).all(axis=1)]
    return data_clean


def fill_outliers(data, method='zscore', axis=0):
    if method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        threshold = 3
        data[z_scores > threshold] = np.nan
        data.fillna(data.median(), inplace=True)
    elif method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        data[(data < lower_bound) | (data > upper_bound)] = np.nan
        data.fillna(data.median(), inplace=True)

    if axis == 0:
        return data
    elif axis == 1:
        return data.T

def drop_outliers(data, method='zscore', axis=0):
    if method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        threshold = 3
        data[z_scores > threshold] = np.nan
        data.dropna(axis=axis, inplace=True)
    elif method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        data[(data < lower_bound) | (data > upper_bound)] = np.nan
        data.dropna(axis=axis, inplace=True)

    return data

# Dataset preview
if st.checkbox("Preview Dataset"):
    if st.button("Head"):
        st.write(data.head())
    elif st.button("Tail"):
        st.write(data.tail())
    else:
        number = st.slider("Select No of Rows", 1, data.shape[0])
        st.write(data.head(number))


# show entire data
if st.checkbox("Show all the data"):
    st.write(data)

st.subheader('To Check Columns Name')
# show column names
if st.checkbox("Show Column Names"):
    st.write(data.columns)

# show dimensions
if st.checkbox("Show the Shape of the dataset"):
    st.write(data.shape)

st.subheader('Summaery of the Data')     
# show summary
if st.checkbox("Show Summary"):
    st.write(data.describe())

numeric_columns = data.select_dtypes(include=['int', 'float'])
st.subheader('Check null values and fill null values ')   
# show missing values
if st.checkbox("Show Missing Values"):
    st.write(numeric_columns.isna().sum())    

# Select a column to treat missing values
col_option = st.multiselect("Select Feature to fillna",numeric_columns.columns)

# Specify options to treat missing values
missing_values_clear = st.selectbox("Select Missing values treatment method", ("Replace with Mean", "Replace with Median", "Replace with Mode"))

if missing_values_clear == "Replace with Mean":
    replaced_value = data[col_option].mean()
    data[col_option]=data[col_option].mean()
    st.write("Mean value of column is :", replaced_value)
elif missing_values_clear == "Replace with Median":
    replaced_value = data[col_option].median()
    st.write("Median value of column is :", replaced_value)
elif missing_values_clear == "Replace with Mode":
    replaced_value = data[col_option].mode()
    st.write("Mode value of column is :", replaced_value)


Replace = st.selectbox("Replace values of column?", ("Yes", "No"))
if Replace == "Yes":
    data[col_option] = data[col_option].fillna(replaced_value,)
    st.write("Null values replaced")
elif Replace == "No":
    st.write("No changes made")

st.subheader(' Check Null values Categorical Columns and fill Null values  ')
#only categorical columns
object_columns = data.select_dtypes(include=['object'])
if st.checkbox("Show Missing Values of object columns"):
    st.write(object_columns.isna().sum()) 

col_option1 = st.multiselect("Select Feature to fillna",object_columns.columns)


# Specify options to treat missing values
missing_values_clear = st.selectbox("Select Missing values For Categorycal columns treatment method", ("Replace with Mean", "Replace with Median", "Replace with Mode"))

if missing_values_clear == "Replace with Mean":
    replaced_value1 = data[col_option1].mean()
    
    st.write("Mean value of column is :", replaced_value1)
elif missing_values_clear == "Replace with Median":
    replaced_value1 = data[col_option1].median()
    st.write("Median value of column is :", replaced_value1)
elif missing_values_clear == "Replace with Mode":
    replaced_value1 ='Missinig'
    
    st.write("Mode value of column is :", replaced_value1)



Replace = st.selectbox("Replace values of column to category?", ("Yes", "No"))
if Replace == "Yes":
    data[col_option1] = data[col_option1].fillna(replaced_value1,)
    st.write("Null values replaced")
elif Replace == "No":
    st.write("No changes made")



if st.checkbox("Show Missing   Values after fill"):
    st.write(data.isna().sum()) 
# To change datatype of a column in a dataframe
# display datatypes of all columns
if st.checkbox("Show datatypes of the columns"):
    st.write(data.dtypes)

st.subheader('Convert Datatype')
col_option_datatype = st.multiselect("Select Column to change datatype", data.columns) 

input_data_type = st.selectbox("Select Datatype of input column", (str,int, float))  
output_data_type = st.selectbox("Select Datatype of output column", (label_encoder,'OneHot_encode'))

st.write("Datatype of ",col_option_datatype," changed to ", output_data_type)
if output_data_type=='OneHot_encode':
    for i in col_option_datatype:
        data = pd.get_dummies(data, columns=[i],drop_first=True)
        
else:
    for i in col_option_datatype:
        data[i] = output_data_type.fit_transform(data[i])
        


if st.checkbox("Show updated datatypes of the columns"):
    st.write(data.dtypes)

if st.checkbox("Preview Dataset aftre convert datatype"):
    if st.button("Head "):
        st.write(data.head())

st.subheader(' Check Outliers and Replace Outliers')
show_outliers = st.checkbox("Show outliers")

# Display data with or without outliers
if show_outliers:
    for k, v in data.items():
            q1 = v.quantile(0.25)
            q3 = v.quantile(0.75)
            irq = q3 - q1
            v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
            perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
            print("Column %s outliers = %.2f%%" % (k, perc))
            st.write(k,perc)


method = st.selectbox("Select the method to fill the outlier ", ("IQR", "Z-score"))

if st.checkbox("Fill Outliers"):
    if method == "IQR":
        data = fill_outliers(data, method='iqr', axis=0)
    elif method == "Z-score":
        data = fill_outliers(data, method='zscore', axis=0)

    st.write("Data with filled outliers")
    st.write(data)

if st.checkbox("Drop Outliers"):
    if method == "IQR":
        data = drop_outliers(data, method='iqr', axis=0)
    elif method == "Z-score":
        data = drop_outliers(data, method='zscore', axis=0)

    st.write("Data with dropped outliers")
    st.write(data)
 

# visualization

st.subheader('Distrubution Plot')

# distribution plot
col = st.selectbox('Which feature to plot?', data.columns)
sns.displot(data[col])
st.pyplot()

st.subheader('Scatter Plot')

# scatter plot
col1 = st.selectbox('Which feature on x?', data.columns)
col2 = st.selectbox('Which feature on y?', data.columns)
fig = px.scatter(data, x =col1,y=col2)
st.plotly_chart(fig)

st.subheader('Correlation Plot') 

# correlartion plots
# Display the correlation heatmap with Seaborn
if st.checkbox("Show Correlation plots with Seaborn"):
    corr = data.corr()
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr, cmap='coolwarm', annot=True, linewidths=0.5, ax=ax1)
    st.pyplot(fig1)

st.subheader('Feature_Scaling')
scaling_method = st.selectbox('Select a scaling method:', ['Standardization', 'Normalization'])

# Perform the selected scaling method on the dataset
if scaling_method == 'Standardization':
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
else:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

# Display the scaled data
st.write('Scaled data:')
st.write(pd.DataFrame(scaled_data, columns=data.columns))

# Machine Learning Algorithms
st.subheader('Machine Learning models')
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

 
 
features = st.multiselect("Select Feature Columns",data.columns)
labels = st.multiselect("select target column",data.columns)

features= data[features].values
labels = data[labels].values


train_percent = st.slider("Select % to train model", 1, 100)
train_percent = train_percent/100

X_train,X_test, y_train, y_test = train_test_split(features, labels, train_size=train_percent, random_state=1)


alg = ['Linear Regression', 'Ridge Regression', 'Lasso Regression']
regression = st.selectbox('Which algorithm?', alg)

if regression == 'Linear Regression':
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    acc = lr.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_lr = lr.predict(X_test)
    mse = mean_squared_error(y_test, pred_lr)
    r2 = r2_score(y_test, pred_lr)
    st.write('Mean Squared Error: ', mse)
    st.write('R2 Score: ', r2)

elif regression == 'Ridge Regression':
    alpha = st.slider('Select alpha value', 0.001, 10.0, 1.0)
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    acc = ridge.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_ridge = ridge.predict(X_test)
    mse = mean_squared_error(y_test, pred_ridge)
    r2 = r2_score(y_test, pred_ridge)
    st.write('Mean Squared Error: ', mse)
    st.write('R2 Score: ', r2)

elif regression == 'Lasso Regression':
    alpha = st.slider('Select alpha value', 0.001, 10.0, 1.0)
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    acc = lasso.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_lasso = lasso.predict(X_test)
    mse = mean_squared_error(y_test, pred_lasso)
    r2 = r2_score(y_test, pred_lasso)
    st.write('Mean Squared Error: ', mse)
    st.write('R2 Score: ', r2)