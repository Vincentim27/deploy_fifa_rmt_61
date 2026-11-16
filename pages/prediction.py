import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

with open('./src/model_lin_reg.pkl', 'rb') as file_1:
  model_lin_reg = pickle.load(file_1)

with open('./src/model_scaler.pkl', 'rb') as file_2:
  model_scaler = pickle.load(file_2)

with open('./src/model_encoder.pkl','rb') as file_3:
  model_encoder = pickle.load(file_3)

with open('./src/list_num_cols.txt', 'r') as file_4:
  list_num_cols = json.load(file_4)

with open('./src/list_cat_cols.txt', 'r') as file_5:
  list_cat_cols = json.load(file_5)

def run():
    st.title('Mari Kita Prediksi Rating Pemain FIFA')
    st.subheader('Silakan lengkapi form di bawah ini:')

    with st.form(key='Form Parameters'):
        Name = st.text_input('Name', value='John Doe') 
        Age = st.slider('Age',min_value=16,max_value=54,value=26,help='Silakan isi usia pemain')
        Height= st.number_input('Height',min_value=155,max_value=206, value=172)
        Weight= st.number_input('Weight',min_value=49,max_value=110,value=70)
        ValueEUR= st.number_input('ValueEUR',value=0)

        st.markdown('---')

        AttackingWorkRate= st.selectbox('AttackingWorkRate',['Low','Medium','High'],index=2)
        DefensiveWorkRate= st.radio('DefensiveWorkRate',['Low','Medium','High'],index=1)

        st.markdown('---')
        PaceTotal= st.number_input('PaceTotal',min_value=0,max_value=100,value=60)
        ShootingTotal= st.number_input('ShootingTotal',min_value=0,max_value=100,value=60)
        PassingTotal= st.number_input('PassingTotal',min_value=0,max_value=100,value=70)
        DribblingTotal= st.number_input('DribblingTotal',min_value=0,max_value=100,value=60)
        DefendingTotal= st.number_input('DefendingTotal',min_value=0,max_value=100,value=60)
        PhysicalityTotal= st.number_input('PhysicalityTotal',min_value=0,max_value=100,value=60)

        submit = st.form_submit_button('Predict')

    data_inf = {
        'Name': Name,
        'Age': Age,
        'Height': Height,
        'Weight': Weight,
        'Price': ValueEUR,
        'AttackingWorkRate': AttackingWorkRate,
        'DefensiveWorkRate': DefensiveWorkRate,
        'PaceTotal': PaceTotal,
        'ShootingTotal': ShootingTotal,
        'PassingTotal': PassingTotal,
        'DribblingTotal': DribblingTotal,
        'DefendingTotal': DefendingTotal,
        'PhysicalityTotal':PhysicalityTotal
    }
    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submit:
        data_inf_num = data_inf[list_num_cols]
        data_inf_cat = data_inf[list_cat_cols]
        # Feature Scaling
        data_inf_num_scaled = model_scaler.transform(data_inf_num)

        # Feature Encoding
        data_inf_cat_encoded = model_encoder.transform(data_inf_cat)

        # Concate
        data_inf_final = np.concatenate([data_inf_num_scaled, data_inf_cat_encoded], axis=1)

        # Predict
        y_pred_inf = model_lin_reg.predict(data_inf_final)
        st.markdown(f'## Prediksi Rating {Name}: {round(y_pred_inf[0],2)}')

if __name__ == '__main__':
  run()