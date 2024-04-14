import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from PIL import Image
# import cv2

# imgg=cv2.imread('6502782.png')
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-color: #fafafa;
opacity: 0.8;
background-image:  linear-gradient(#f7ff00 2.4000000000000004px, transparent 2.4000000000000004px), linear-gradient(90deg, #f7ff00 2.4000000000000004px, transparent 2.4000000000000004px), linear-gradient(#f7ff00 1.2000000000000002px, transparent 1.2000000000000002px), linear-gradient(90deg, #f7ff00 1.2000000000000002px, #fafafa 1.2000000000000002px);
background-size: 60px 60px, 60px 60px, 12px 12px, 12px 12px;
background-position: -2.4000000000000004px -2.4000000000000004px, -2.4000000000000004px -2.4000000000000004px, -1.2000000000000002px -1.2000000000000002px, -1.2000000000000002px -1.2000000000000002px;
}

</style> 
"""
st.markdown(page_bg_img,unsafe_allow_html=True)

# img=Image.open('imagescard.png')
img1=Image.open('6502782.png')
# st.image(
#     img,
#     width=300,
#     channels='rgb'
# )
st.image(
    img1,
    width=300,
    channels='rgb'
)


testmodel=pickle.load(open("bank_loan_model.sav","rb"))

def bank_loan_pred(input_data):
    input_np=np.asarray(input_data)
    input_np1=input_np.astype(int)
    input_re=input_np1.reshape(1,-1)
    # std_=sc.transform(input_re)
    pred=testmodel.predict(input_re)
    print(pred)
    if (pred==0):
        return 'the loan might be approve !!!'
    else:
        return 'the loan might not be approve !!!'

def main():
    st.title('BANK LOAN APPROVAL') 
    
    no_of_dependents=st.text_input(' no_of_dependents')
    education=st.text_input(' education')
    self_employed=st.text_input(' self_employed')
    income_annum=st.text_input(' income_annum')
    loan_amount=st.text_input('loan_amount')
    loan_term=st.text_input('loan_term')
    cibil_score=st.text_input('cibil_score')
    residential_assets_value=st.text_input('residential_assets_value')
    commercial_assets_value=st.text_input('commercial_assets_value')
    luxury_assets_value=st.text_input('luxury_assets_value')
    bank_asset_value=st.text_input('bank_asset_value')

    loan_status=''
    if st.button('bank loan approval'):
        loan_status=bank_loan_pred([no_of_dependents,education,self_employed,income_annum,loan_amount,loan_term,cibil_score,residential_assets_value,commercial_assets_value,luxury_assets_value,bank_asset_value])
    st.success(loan_status)

if __name__== '__main__':
    main()
