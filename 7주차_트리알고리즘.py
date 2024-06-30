#!/usr/bin/env python
# coding: utf-8

# 1. 중고차의 가격을 예측하는 선형 회귀모델을 만들기
# - Price, Power 변수 이용
# - 결측값 제거  
# 1.1. 모델의 그래프를 그려보고 평가하기.  
# 1.2. 450의 power을 가진 자동차의 price를 예측하기.  
# 2. 중고차의 가격 등급(Price_Rank, 1이 가장 비싼 가격의 그룹임)를 예측하는 로지스틱회귀 모델을 만들고 평가해보기.
# - 변수는 가격, 연비, 연식 등 조별로 선택.  
# 
# [데이터 설명]  
# - Name: 브랜드명
# - Location: 자동차 판매 도시
# - Year: 자동차 제조 연도
# - Kilometers_Driven: 주행 킬로미터(이전 소유자가 주행한 총 킬로미터)
# - Fuel_Type: 자동차가 사용한 연료 종류
# - Transmission: 자동차 변속기 종류(자동/수동)
# - Owner_Type: 자동차 주인 변경 횟수(first: 처음 중고차로 나옴)
# - Mileage: 연비
# - Engine: cc단위 엔진 배기량
# - Power : 엔진 최대 출력
# - Seats: 좌석 수
# - Price_Rank: 가격대 랭킹(1-4, 1이 가장 비싼 그룹)
# - Price: 중고차 가격

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv('car_data.csv')
df.info()


# In[2]:


#단위 제거, float타입으로 변환하기
df['Mileage'] = df['Mileage'].str.split().str[0].astype('float')
df['Engine'] = df['Engine'].str.split().str[0].astype('float')
df['Power'] = df['Power'].str.split().str[0].astype('float')
df.head()


# In[3]:


#Mileage 0인 값 제거하기
df.drop(df[df['Mileage'] == 0].index, inplace = True)
df[df['Mileage'] == 0]


# In[4]:


#범주형 데이터들은 0, 1, 2로 바꾸기
df['Fuel_Type'].replace({'Petrol' : 0, 'Diesel' : 1, 'CNG' : 2}, inplace = True)
df['Transmission'].replace({'Automatic' : 0, 'Manual' : 1}, inplace = True)
df['Owner_Type'].replace({'First' : 0, 'Second' : 1, 'Third' : 2}, inplace = True)
df.head()


# In[5]:


df.drop(df[['Name','Location']], axis = 1, inplace = True)
df.head()


# In[6]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
price = df[['Price']]
power = df[['Power']]


# In[ ]:


import matplotliv.pyplot as plt
plt.scatter(power, price)
plt.xlabel('Power')
plt.ylabel('Price')
plt.show


# In[ ]:


from sklearn.model_selection import train_test_split
train_lrx, test_lrx,train_lry, test_lry = train_test_split(power, price, random_state = 42)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_lrx,train_lry)


# In[ ]:


print(lr.score())


# In[7]:


import numpy as np
input_data = df[['Year', 'Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage','Engine','Power','Seats']].to_numpy()
target_data = df['Price_Rank'].to_numpy()


# In[8]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
train_x, test_x,train_y,test_y = train_test_split(input_data, target_data, test_size = 0.2, random_state = 10)
lr = LogisticRegression(multi_class='multinomial')


# In[9]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

# 데이터를 모델에 학습
rf.fit(train_x, train_y)

# 각 특성의 중요도 확인
importances = rf.feature_importances_

# 중요도를 기준으로 특성의 순서를 정렬
indices = np.argsort(importances)[::-1]

print(df[['Year', 'Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage','Engine','Power','Seats']].columns[indices])
print(importances)


# ### 특성 중요도 파악하기

# In[10]:


import matplotlib.pyplot as plt

# 중요도를 시각화
plt.figure(figsize=(10, 6))
plt.bar(range(train_x.shape[1]), importances[indices])
plt.xticks(range(train_x.shape[1]), df[['Year', 'Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage','Engine','Power','Seats']].columns[indices], rotation=90)
plt.title("Features Importance")
plt.show()


# In[11]:


params = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter':[1,10,100,1000,10000]}
grid_search = GridSearchCV(lr, params, cv=5)
grid_search.fit(train_x, train_y)


# In[12]:


print(grid_search.score(train_x,train_y))
print(grid_search.score(test_x,test_y))


# ## 수치형 데이터만 써보자. 어차피 중요도에서 수치형데이터들이 앞쪽임
# 수치형 데이터들만 쓰기 때문에 편하게 표준화해보자

# In[13]:


input_data = df[['Year', 'Kilometers_Driven','Mileage','Engine','Power']].to_numpy()
target_data = df['Price_Rank'].to_numpy()


# In[14]:


from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(input_data, target_data, random_state=42)


# In[15]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


# In[16]:


params = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter':[1,10,100,1000,10000]}
grid_search = GridSearchCV(lr, params, cv=5)
grid_search.fit(train_scaled, train_target)


# In[17]:


print(grid_search.score(train_scaled,train_target))
print(grid_search.score(test_scaled,test_target))


# In[18]:


grid_search.best_params_


# #### 기본적으로 아무것도 안 한 모델보다는 높은 예측률을 보여줌
