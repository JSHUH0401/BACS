#!/usr/bin/env python
# coding: utf-8

# ## [실습]
# ```python
# 1. 당뇨병 데이터 분석해보기
# 2. 데이터 셋 분할 후 표준점수 활용해 전처리 하기
#     2.1. 클래스 간 가설 설정 후 검증해보기 (t-test 활용)
# 3. KNeighborsClassifier() 적용해보기
#     3.1. 다양한 알고리즘 활용 가능
# ```
# ## [데이터 설명]
# 
# Pregnancies	임신 횟수 - *비율척도*\
# Glucose 혈당 - *비율척도*\
# BloodPressure 혈압 - *비율척도*\
# SkinThickness 피부 두께 - *비율척도*\
# Insulin 인슐린 - *비율척도*\
# BMI 체질량 지수 - *비율척도*\
# DiabetesPedigreeFunction 당뇨병 혈통 함수 - *비율척도*\
# Age 나이 - *비율척도*\
# Outcome 당뇨병 유무 - *명목척도*\
# -당뇨병 없음: 0\
# -당뇨병 있음: 1

# In[1]:


import pandas as pd


# In[2]:


d = pd.read_csv('diabetes.csv')
d.head()


# In[3]:


# 관련 패키지 설치
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


d.dtypes


# In[5]:


# 데이터 결측치 확인
print(d.isnull().sum())


# In[6]:


for column in d.columns[[1,2,3,4,5]]:
    d[column].replace(0,d[column].mean(), inplace = True)


# In[7]:


# 독립변수와 종속변수 나눠주기
X = d.iloc[:,0:8].values
Y = d.iloc[:, -1].values


# In[8]:


dh = d.hist(figsize = (15, 15))


# In[9]:


# 상관관계 시각화 하기

corr = sns.heatmap(d.corr(), annot=True)
plt.show()


# In[10]:


# 사이킷런 패키지 설치
from sklearn.model_selection import train_test_split


# In[11]:


# 훈련모델과 테스트 모델 나눠주기
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)


# In[12]:


from sklearn.preprocessing import StandardScaler
ss =StandardScaler()
ss.fit(X_train)
train_scaled = ss.fit_transform(X_train)
test_scaled = ss.fit_transform(X_test)


# In[13]:


from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_scaled, y_train)

# 모델 학습
kn.fit(train_scaled, y_train)

# permutation importance 계산
results = permutation_importance(kn, test_scaled, y_test, n_repeats=100)

# permutation importance 시각화
sorted_idx = np.argsort(results['importances_mean'])
plt.barh(d.columns[sorted_idx], results['importances_mean'][sorted_idx])
plt.show()


# In[14]:


print(kn.score(train_scaled, y_train))
print(kn.score(test_scaled, y_test))


# In[15]:


from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(train_scaled, y_train)

# permutation importance 계산
impor = {'BMI':0,'SkinThickness':0,'Pregnancies':0,'BloodPressure':0,'DiabetesPedigreeFunction':0,'Insulin':0,'Age':0,'Glucose':0}
for _ in range(0,10):
    results = permutation_importance(kn, test_scaled, y_test, n_repeats=100)
    sorted_idx = np.argsort(results['importances_mean'])
    top_3 =  list(d.columns[sorted_idx[-3:]])
    for i in top_3:
        impor[i] += 1
impor


# ### BMI, Age, Glucose를 특성으로 RandomForestClassifer 수행

# In[16]:


input_data = d[['BMI','Age','Glucose']].to_numpy()
target_data = d['Outcome'].to_numpy()


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, random_state=42)


# #### 표준화 안 한 경우

# In[18]:


from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(X_train, y_train)
print(kn.score(X_train,y_train))
print(kn.score(X_test,y_test))


# #### 표준화 진행한 경우

# In[19]:


from sklearn.preprocessing import StandardScaler
ss =StandardScaler()
ss.fit(X_train)
train_scaled = ss.fit_transform(X_train)
test_scaled = ss.fit_transform(X_test)


# In[20]:


from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_scaled, y_train)
print(kn.score(train_scaled,y_train))
print(kn.score(test_scaled,y_test))

