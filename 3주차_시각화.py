#!/usr/bin/env python
# coding: utf-8

# ### 따릉이의 대여 데이터를 보면서 대여수에 가장 큰 영향을 주는 기상조건을 알아보자

# ## 데이터 전처리

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('따릉이.csv')
data.head()


# In[2]:


#데이터 속 결측치 개수 확인
data.isna().sum()


# In[3]:


#hour_bef_precipitation 이 Null값인 행 삭제. 이건 평균값으로 할 수가 없음, Null인 행 2개뿐
data = data[~data['hour_bef_precipitation'].isna()]


# In[4]:


#각 칼럼의 평균값으로 널값을 대체
for col in data.columns:
    data.replace({col: np.nan}, data[col].mean(), inplace = True)
data.isna().sum()


# ## 데이터 확인하기
# #### 'id'위치 데이터이기 때문에 날씨가 아닌 위치에 따라 대여수가 영향을 받을 수 있음.
# #### 따라서 'hour'에 따라 'count'가 늘어나는지 확인 필요.
# ##### - 비례O 위치의 개입이 적음
# ##### - 비례X 위치의 개입이 큼

# In[5]:


# 'hour'과 'count'의 비례관계 확인하기
h_c = data.groupby('hour').mean().reset_index()
plt.plot(h_c['hour'],h_c['count'])


# #### 비례하지 않는다고 판단
# #### >> 시각화했을 때, 확실한 요인(강수처럼)이 아니라면 들쭉날쭉할지도?

# ### 대여량 비교는 시간당 대여수(count / hour)로 파악

# In[6]:


# 0으로 카운트를 나눌 수 없어서 0인 hour는 0.5로 대체함.
# 데이터에 'per hour'을 추가
data.replace({'hour': 0}, 0.5, inplace = True)
data['per_hour'] =data['count'].values / data['hour'].values
data.head()


# ### 각 항목(미세먼지, 풍속, 강수)에 대한 데이터 생성

# In[7]:


#풍속, 미세먼지(2.5, 10), 강수를 분리해서 각 대여수를 모두 더함. ex) 풍속이 2.0일 때 시간당 대여수를 모두 더함.
windspeed_data = data.groupby('hour_bef_windspeed').sum().reset_index()
pm2_data = data.groupby('hour_bef_pm2.5').sum().reset_index()
pm10_data = data.groupby('hour_bef_pm10').sum().reset_index()
precipitation_data = data.groupby('hour_bef_precipitation').sum().reset_index()
pm10_data.head()


# In[8]:


plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['font.family']


# ## 미세먼지 시각화

# In[9]:


plt.plot(pm10_data['hour_bef_pm10'], pm10_data['per_hour'], alpha = 0.5)
plt.plot(pm2_data['hour_bef_pm2.5'], pm2_data['per_hour'], c = 'r', alpha = 0.5)
# 이상치, 축의 범위 쓸데없이 넓음.


# In[10]:


#이상치 제거
pm10_data.drop(index = pm10_data[pm10_data['per_hour'] == max(pm10_data['per_hour'])].index, axis = 1, inplace = True)
pm2_data.drop(index = pm2_data[pm2_data['per_hour'] == max(pm2_data['per_hour'])].index, axis = 1, inplace = True)


# ### 미세먼지 시각화

# In[11]:


#시각화 전, 크기설정 및 서브플롯 생성
fig, ax = plt.subplots()
plt.rcParams["figure.figsize"] = (10,5)

#미세먼지 시각화(pm10, pm2.5)
ax.plot(pm10_data['hour_bef_pm10'], pm10_data['per_hour'], alpha = 1, linewidth = 2)
ax.plot(pm2_data['hour_bef_pm2.5'], pm2_data['per_hour'], c = 'r', alpha = 0.5, linewidth =2)
ax.set_ylim([0,1500])             #y축 범위제한
ax.set_xlim([0, 200])             #x축 범위제한
ax.set_xlabel('미세먼지농도')     #x축 제목
ax.set_ylabel('대여수')           #y축 제목
ax.legend(['pm10', 'pm2.5'])  #범례표시
ax.grid(alpha = 0.5)          #격자
ax.set_title('미세먼지에 따른 대여수', fontsize = 10)

#각 그래프 별 최댓값, 최솟값 표시하기
#각 그래프의 최댓값, 최솟값을 보유한 행의 인덱스 추출.
pm10_max = np.argmax(pm10_data['per_hour'])
pm10_min = np.argmin(pm10_data['per_hour'])
pm2_max = np.argmax(pm2_data['per_hour'])
pm2_min = np.argmin(pm2_data['per_hour'])

#최댓값, 최솟값에 마크 생성
ax.plot(pm10_data['hour_bef_pm10'][pm10_max], pm10_data['per_hour'][pm10_max], 'o', ms=5, c = 'b')
ax.plot(pm10_data['hour_bef_pm10'][pm10_min], pm10_data['per_hour'][pm10_min], 'o', ms=5, c = 'b')
ax.plot(pm2_data['hour_bef_pm2.5'][pm2_max], pm2_data['per_hour'][pm2_max], 'o', ms=5, c = 'r')
ax.plot(pm2_data['hour_bef_pm2.5'][pm2_min], pm2_data['per_hour'][pm2_min], 'o', ms=5, c = 'r')

#최댓값, 최솟값의 레이블 표시
for spot in [pm10_max, pm10_min]:
    ax.text(pm10_data['hour_bef_pm10'][spot], pm10_data['per_hour'][spot] + 20, '('+str(round(pm10_data['hour_bef_pm10'][spot],2))+', '+str(round(pm10_data['per_hour'][spot],2))+')',
             fontsize = 10,                 # 글자 크기
             fontweight = 'bold',           # 글자 굵기 - 볼드체
             color = 'black',               # 글자 색
             horizontalalignment = 'center',# 글자 배치
             verticalalignment = 'bottom')
for spot in [pm2_max, pm2_min]:
    ax.text(pm2_data['hour_bef_pm2.5'][spot], pm2_data['per_hour'][spot] + 20,'('+str(round(pm2_data['hour_bef_pm2.5'][spot],2))+', '+str(round(pm2_data['per_hour'][spot],2))+')',
             fontsize = 10,                 # 글자 크기
             fontweight = 'bold',           # 글자 굵기 - 볼드체
             color = 'black',               # 글자 색
             horizontalalignment = 'center',# 글자 배치
             verticalalignment = 'bottom')


# ### 풍속 시각화
# 

# In[12]:


fig, ax = plt.subplots(1,2)
plt.rcParams["figure.figsize"] = (10,4)

a = ax[0].scatter(windspeed_data['hour_bef_windspeed'],windspeed_data['per_hour'], s=150,  edgecolors="black", c = windspeed_data["hour_bef_windspeed"], alpha=0.3, cmap="PuBu")
ax[0].set_xlabel('풍속')
ax[0].set_ylabel('대여수')
ax[0].set_title('풍속에 따른 대여수', fontsize = 10)
ax[0].grid( True, axis = 'x')

ax[1].bar(windspeed_data['hour_bef_windspeed'], windspeed_data['per_hour'], color = 'g', width = 0.2)
ax[1].set_xlabel('풍속')
ax[1].set_ylabel('대여수')
ax[1].set_title('풍속에 따른 대여수', fontsize = 10)
ax[1].grid( True, axis = 'x')


# ### 강수 시각화

# In[13]:


pathces, texts, autotexts = plt.pie(precipitation_data['per_hour'], labels = ['강수 X', '강수 O'], 
                                    startangle = 45,autopct = '%.1f%%', shadow=True)
for text in texts:
    text.set_fontsize(10)
    text.set_color('black')
    
for text in autotexts:
    text.set_fontsize(10)
    text.set_fontweight('bold')
    text.set_color('white')
plt.title('강수 여부에 따릉이 대여 비율', fontsize=10)

plt.subplots_adjust(left=0.125, bottom=0.1,  right=0.9, top=0.9, wspace=0.4, hspace=0.35)


# In[18]:


#시각화 전, 크기설정 및 서브플롯 생성
fig, ax = plt.subplots(1,3)
plt.rcParams["figure.figsize"] = (10,3)

#미세먼지 시각화(pm10, pm2.5)
ax[0].plot(pm10_data['hour_bef_pm10'], pm10_data['per_hour'], alpha = 1, linewidth = 2)
ax[0].plot(pm2_data['hour_bef_pm2.5'], pm2_data['per_hour'], c = 'r', alpha = 0.5, linewidth =2)
ax[0].set_ylim([0,1500])             #y축 범위제한
ax[0].set_xlim([0, 200])             #x축 범위제한
ax[0].set_xlabel('미세먼지농도')     #x축 제목
ax[0].set_ylabel('대여수')           #y축 제목
ax[0].legend(['pm10', 'pm2.5'])  #범례표시
ax[0].grid(alpha = 0.5)          #격자
ax[0].set_title('미세먼지에 따른 대여수', fontsize = 10)

#각 그래프 별 최댓값, 최솟값 표시하기
#각 그래프의 최댓값, 최솟값을 보유한 행의 인덱스 추출.
pm10_max = np.argmax(pm10_data['per_hour'])
pm10_min = np.argmin(pm10_data['per_hour'])
pm2_max = np.argmax(pm2_data['per_hour'])
pm2_min = np.argmin(pm2_data['per_hour'])

#최댓값, 최솟값에 마크 생성
ax[0].plot(pm10_data['hour_bef_pm10'][pm10_max], pm10_data['per_hour'][pm10_max], 'o', ms=5, c = 'b')
ax[0].plot(pm10_data['hour_bef_pm10'][pm10_min], pm10_data['per_hour'][pm10_min], 'o', ms=5, c = 'b')
ax[0].plot(pm2_data['hour_bef_pm2.5'][pm2_max], pm2_data['per_hour'][pm2_max], 'o', ms=5, c = 'r')
ax[0].plot(pm2_data['hour_bef_pm2.5'][pm2_min], pm2_data['per_hour'][pm2_min], 'o', ms=5, c = 'r')

#최댓값, 최솟값의 레이블 표시
for spot in [pm10_max, pm10_min]:
    ax[0].text(pm10_data['hour_bef_pm10'][spot], pm10_data['per_hour'][spot] + 20, '('+str(round(pm10_data['hour_bef_pm10'][spot],2))+', '+str(round(pm10_data['per_hour'][spot],2))+')',
             fontsize = 10,                 # 글자 크기
             fontweight = 'bold',           # 글자 굵기 - 볼드체
             color = 'black',               # 글자 색
             horizontalalignment = 'center',# 글자 배치
             verticalalignment = 'bottom')
for spot in [pm2_max, pm2_min]:
    ax[0].text(pm2_data['hour_bef_pm2.5'][spot], pm2_data['per_hour'][spot] + 20,'('+str(round(pm2_data['hour_bef_pm2.5'][spot],2))+', '+str(round(pm2_data['per_hour'][spot],2))+')',
             fontsize = 10,                 # 글자 크기
             fontweight = 'bold',           # 글자 굵기 - 볼드체
             color = 'black',               # 글자 색
             horizontalalignment = 'center',# 글자 배치
             verticalalignment = 'bottom')
    
    
#풍속 시각화    
ax[1].bar(windspeed_data['hour_bef_windspeed'], windspeed_data['per_hour'], color = 'g', width = 0.2)
ax[1].set_xlabel('풍속')
ax[1].set_ylabel('대여수')
ax[1].set_title('풍속에 따른 대여수', fontsize = 10)
ax[1].grid( True, axis = 'x')

#강수여부 시각화
pathces, texts, autotexts = ax[2].pie(precipitation_data['per_hour'], labels = ['강수 X', '강수 O'], 
                                    startangle = 45,autopct = '%.1f%%', shadow=True)
for text in texts:
    text.set_fontsize(10)
    text.set_color('black')
    
for text in autotexts:
    text.set_fontsize(10)
    text.set_fontweight('bold')
    text.set_color('white')
ax[2].set_title('강수 여부에 따릉이 대여 비율', fontsize=10)

plt.subplots_adjust(left=0.125, bottom=0.1,  right=0.9, top=0.9, wspace=0.4, hspace=0.35)


# ### 결론
# #### 생각보다 상식적인 결과가 나옴
# #### 많은 데이터 수가 위치의 영향을 최소화했다고 결론.
