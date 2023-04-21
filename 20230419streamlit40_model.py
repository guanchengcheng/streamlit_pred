import glob
import pickle
import pandas as pd
import numpy as np
import streamlit as st
#import matplotlib.pyplot as plt
#from PIL import Image
#from unittest import result
#import shap

st.set_page_config(layout="wide", page_title="Prognostic Models in Critically Ill Patients with Sepsis-associated Acute Kidney Injury")

st.sidebar.image("R.jpg")

s1 = ["status 7d",'status 14d','status 28d']
s2 = ["_7", "_14", "_28"]
status = {i:j for i, j in zip(s1, s2)}
st.sidebar.info("**Status & model select**")
data = st.sidebar.selectbox("Choose your status", s1)
models = ["LogisticRegression", "MLPClassifier", "RandomForestClassifier", "SVC", "XGBClassifier"]
model_type = st.sidebar.selectbox("Choose your model", models, 4)

st.sidebar.warning('**Here, we recommend XGBoost model for prediction.**')
st.sidebar.warning('The remaining numerical variables were values during the first 24 hours of ICU admission. ')
st.sidebar.warning('For the feature of AKI-stage Ⅲ, 0 represents no disease; 1 means having the disease.')


st.info('#### Prognostic Models in Critically Ill Patients with Sepsis-associated Acute Kidney Injury')

model = []
for f in s2:
    fname = []
    for i in glob.glob("*"):
        if f in i:
           fname.append(i)
    model.append(fname)

file = None
no_features = ['status_28d','status_14d', 'status_7d', 'survival_time']

for i in model:
    if status[data] in i[0]:
        file = i

for m in file:
    if model_type in m:
        M = m

# 参数设置部分
val = pd.read_excel("变量范围.xlsx").T
v = val.index.tolist()
val.fillna("", inplace=True)

with open(M, 'rb') as fmodel:
    model = pickle.load(fmodel)

with st.expander("**Params setting**", True):
    k = 0
    col = st.columns(5)
    for i in range(1, val.shape[0]):
        if (val.iloc[i][3] !="") and (v[i] in model.feature_names_in_):
            st.session_state[v[i]] = col[k%5].number_input(v[i].replace("_", " ")+"("+val.iloc[i][3]+")",
                                                             min_value=float(val.iloc[i][0]),
                                                             max_value=float(val.iloc[i][1]),
                                                             step=float(val.iloc[i][2]))
            k = k+1
        elif v[i] in model.feature_names_in_:
            st.session_state[v[i]] = col[k%5].number_input(v[i].replace("_", " "),
                                                             min_value=float(val.iloc[i][0]),
                                                             max_value=float(val.iloc[i][1]),
                                                             step=float(val.iloc[i][2]))
            k = k+1

    col1 = st.columns(5)
    start = col1[2].button("Start predict", use_container_width=True)

if start:
    model_name = None
    for i in file:
        if model_type in i:
            model_name = i

    with open(model_name, 'rb') as fmodel:
        model = pickle.load(fmodel)
        X = np.array([[st.session_state[i] for i in model.feature_names_in_]])
        with st.expander("**Current parameters and predict result**", True):
            p = pd.DataFrame([{i:st.session_state[i] for i in model.feature_names_in_}])
            p.index = ["params"]
            st.write(p)

            y_pred = model.predict(X)
            y_pred_prob = model.predict_proba(X)
            res = y_pred[0]
            res = 'Survival' if res else 'Non-survival'
            
            st.success(f"The prediction is successful. The result of **{data}** with model **{model_type}** is **{res}**, probability is following:")

            pred_prob = pd.DataFrame([[round(y_pred_prob[0][0], 3), round(y_pred_prob[0][1], 3)]], columns=['Survival','Non-survival'])
            pred_prob.index = ["pred prob"]
            st.dataframe(pred_prob, use_container_width=True)
else:
    with st.expander("**Current parameters and predict result**", True):
        st.warning("**No models are currently used for prediction!**")

check_raw = st.sidebar.columns(2)

raw_data = check_raw[0].checkbox('Show raw data')
use_info = check_raw[1].checkbox('Show use info')

if use_info:
    # 展示使用教程
    with st.expander("**Use info**", True):
        st.text('1. In left sidebar menu, select your status and model;')
        st.text('2. In main window, set your params in params set part;')
        st.text('3. Click \'start predict\' button to predict your input data with your status and model;')
        st.text('4. In result show part, you can get predict result.')
        st.write('**Otherwise, you can show raw data by select sidebar \'Show raw data\' checkbox.**')
        

if raw_data:
    # 展示原始数据部分
    with st.expander("**Raw data**", True):
        disp=pd.read_csv('2499_displayed.csv', encoding="utf-8")
        st.dataframe(disp, use_container_width=True)

    # 训练数据文件
    df1=  pd.read_csv(file[0], encoding="utf-8")

    with st.expander("**Train data sets**", True):
        st.dataframe(df1, use_container_width=True)
