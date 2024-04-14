
import numpy as np
import pickle


# In[3]:


testmodel=pickle.load(open(r'C:\Users\AJ\bankloan\bank_loan_model.sav','rb'))


# In[4]:


input=(0,1,0,4100000,12200000,8,417,2700000,2200000,8800000,3300000)
input_np=np.asarray(input)
input_re=input_np.reshape(1,-1)
# std_=sc.transform(input_re)
pred=testmodel.predict(input_re)
if (pred==0):
    print('the loan might be approve !!!')
else:
    print ('the loan might not be approve !!!')