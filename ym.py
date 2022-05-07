# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:08:34 2022

@author: Zhou N
"""


import numpy as np 
import pandas as pd



from tensorflow.keras.models import load_model

ann = load_model('heart.h5')



import streamlit as st
from PIL import Image
image = Image.open('image_1.png')


col1, col2=st.columns([2,3])

with st.sidebar:
    st.title('心脏病风险预测')   
    st.image(image)
    st.subheader('基本信息')
    st.text_input('姓名')
    age=st.number_input('年龄')
    gender=st.selectbox('性别',options=('男','女'))
    if gender=='男':
        sex=1
    else:
        sex=0
    
    
with col1:
    st.subheader('健康信息区')
    cp=st.selectbox('是否胸痛',options=(0,1,2,3))
    trestbps=st.number_input('静息血压')
    chol=st.number_input('血脂')
    fb=st.selectbox('血糖值是否大于120mg/dl',options=('是','否'))
    if fb=='是':
        fbs=1
    else:
        fbs=0
with col1:   
    thalach=st.number_input('血糖值')
    restecg=st.selectbox('静息心电图结果',options=(0,1))   
    
with col1:
    oldpeak=st.number_input('运动后ST段降低')
    slope=st.selectbox('峰值运动时ST段坡度',options=(0,1,2))
    exang=st.selectbox('器官或部位长期疼痛',options=(0,1))
    ca=st.selectbox('肿瘤标志物',options=(0,1))
    tha=st.selectbox('影像结果',options=('正常','可逆损伤','不可逆损伤'))
    if tha=='正常':
        影像结果=1
    elif tha=='可逆损伤':
        影像结果=2
    else:
        影像结果=3



#八、将所收集数据构成DataFrame
allfactor=[[age,sex, cp,trestbps,chol,fbs,restecg,thalach,
       exang,oldpeak,slope,ca,thal]]

allfactor=pd.DataFrame(allfactor)

#九、调用模型预测
output=ann.predict(allfactor)
    
#十、定义函数解读预测结果
def dis(output):
    if output==1:
        return '您患心脏病风险较高'
    else:
        return '您患心脏病风险较低'

#十一、输出
with col2:
    st.subheader('预测结果')
    outcome=dis(output)
    st.write('预测结果为',outcome)  
    st.subheader('未来5年心脑血管病趋势预测')  
    vessels=np.random.rand(5,2) 
    line_data = pd.DataFrame(
    vessels,
    columns=['心脏病风险', '卒中风险'])
   
    st.line_chart(line_data,use_container_width=True,height=240)   

    st.subheader('癌前病变风险评估') 
    x = ('黏膜白斑','腺瘤性肠息肉','慢性萎缩性胃炎','乳腺囊性增生','肝硬化')
    y = np.random.rand(5,1) 
    chart_data= pd.DataFrame(
        y,index=x,columns=['风险程度'])
    st.bar_chart(chart_data,use_container_width=True,height=240)


    
