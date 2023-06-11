from transformers import pipeline
import json
import nltk
from nltk import word_tokenize
import streamlit as st
import os

def load_data(uploaded_file):
    data = []
    if uploaded_file is not None:
        data = json.load(uploaded_file)
    return data

def tokenize(sent):
    tokens = ' '.join(word_tokenize(sent.lower()))
    return tokens

def clean_data(text):
    text = text.replace('{ vocalsound } ', '')
    text = text.replace('{ disfmarker } ', '')
    text = text.replace('a_m_i_', 'ami')
    text = text.replace('l_c_d_', 'lcd')
    text = text.replace('p_m_s', 'pms')
    text = text.replace('t_v_', 'tv')
    text = text.replace('{ pause } ', '')
    text = text.replace('{ nonvocalsound } ', '')
    text = text.replace('{ gap } ', '')
    return text

def prepare_data(data,query):
    entire_src = []
    for i in range(len(data)):
        cur_turn = data[i]['speaker'].lower() + ': '
        cur_turn = cur_turn + tokenize(data[i]['content'])
        entire_src.append(cur_turn)
    entire_src = ' '.join(entire_src)
    prepared_data = clean_data('<s> ' + query + ' </s> ' + entire_src + ' </s>')
    return prepared_data

def predict(selected_model, prepared_data):
    predicted = selected_model(prepared_data)
    return predicted

def submit_check(uploaded_file,query,model_name,sum_nocut,sum_gen):
    data = []
    # use_demo = st.radio(
    #     "choose input type",
    #     ('use demo meeting transcript','your own file'))
    # if st.button('submit'):
    #     st.write('submitted')
    #     if use_demo == 'use demo meeting transcript':
    #         file_name = 'test_data/1_meet.json'
    #         with open(file_name) as f:
    #             uploaded_file = f
    #     else:
        data = load_data(uploaded_file)
        prepared_data = prepare_data(data,query)
        if model_name == 'fgiuhsdfkjhfv/longsec_withno_cut':
            predicted = predict(sum_nocut, prepared_data)
        else :
            predicted = predict(sum_gen, prepared_data)
        show_output(predicted)

def show_output(predicted):
    st.write('predict', predicted[0]['summary_text'])

def main():
    print(os.listdir('.'))
    sum_nocut = pipeline(task="summarization",model='fgiuhsdfkjhfv/longsec_withno_cut')
    sum_gen = pipeline(task="summarization",model='fgiuhsdfkjhfv/longsec_general_query')

    uploaded_file = st.file_uploader("choose meeting json", type=['json'])
    model_name = st.selectbox(
        'select model',
        ('fgiuhsdfkjhfv/longsec_withno_cut', 'fgiuhsdfkjhfv/longsec_general_query'))
    query = st.text_input('input query', 'summarize this meeting')
    submit_check(uploaded_file,query,model_name,sum_nocut,sum_gen)

if __name__ == "__main__": 
    main()