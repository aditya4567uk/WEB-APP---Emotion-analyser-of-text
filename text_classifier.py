# -*- coding: utf-8 -*-
"""
Created on Feb 24 21:09:52 2022

@author: adity
"""

import pandas as pd


import streamlit as st

import joblib

import altair as alt

pipe_lr = joblib.load(open("C:\\Users\\adity\\Desktop\\html\\projectEX2\\emotion_classifier_model.pkl","rb"))

def predict_emotions(docx):
    results=pipe_lr.predict([docx])
    return results

def get_prediction_proba(docx):
    results=pipe_lr.predict_proba([docx])
    return results


def main():
    st.title("Text-Analiser")
    menu=["Home","Monitor","About"]
    choice=st.sidebar.selectbox("Menu",menu)
    
    if choice =="Home":
        st.subheader("Home-Emotion in text")
        with st.form(key='emotion_clf_form'):
            raw_text=st.text_area("Type Here")
            textlen=len(raw_text)
            space=" "
            i=0
            while i<textlen-1:
                if raw_text[i]==space:
                    c=i+1
                    if raw_text[c]==space:
                        exp_text1="Double Space or more found ,result might not be accurate"
                        st.write(exp_text1)
                i=i+1
            if len(raw_text)==0:
                exp_text3="Empty String found Hence emotions are Neutral"
                st.write(exp_text3)
            if raw_text.isdigit():
                exp_text2="contains only digit , can't show sensible results"
                st.write(exp_text2)
            if len(raw_text)==1:
                exp_text4="Found only 1 character ,can't show sensible results or consider it as neutral"
                st.write(exp_text4)
                    
            submit_text = st.form_submit_button(label='Submit')
            if submit_text:
                col1,col2 = st.columns(2)
                prediction = predict_emotions(raw_text)
                probability = get_prediction_proba(raw_text)
                with col1:
                    st.success("Original Text")
                    st.write(raw_text)
                    
                    st.success("Prediction Probablity")
                    
                    
                    proba_df=pd.DataFrame(probability,columns=pipe_lr.classes_)
                    st.write(proba_df.T)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions","probability"]
                    

                with col2:
                    st.success("Prediction")
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions","probability"]
                    
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
                    st.altair_chart(fig,use_container_width=True)
    elif choice =="Monitor":
        st.subheader("Monitor App")
    else:
        st.subheader("About")
        about_text="Contributors --->  20MIP10001 - Anush Dubey || 20MIP10002 - Akanksha Verma || 20MIP10008 - Aditya Kumar || 20MIP10027 - Abhishek Kushwaha || 20MIP10035 - Ishika Shrivastava"
        st.write(about_text)

if __name__ == '__main__':
    main()