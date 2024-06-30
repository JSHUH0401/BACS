#!/usr/bin/env python
# coding: utf-8

# ### 2주차 실습 데이터: 'World-Stock-Prices-Dataset.csv' /미국 주식시장 주가 데이터(2000.01.01~2023.09.18)  
# 1. 조 별로 산업 선정 (각 산업의 종목 수를 비교해서 3개 이상 종목을 보유한 산업을 선정)  
# 2. 6개월(2023-03-18~2023-09-18), 종가 기준으로 수익률이 가장 높은 종목과 가장 낮은 종목 확인  
# 3. 각 종목의 수익률을 시각화

# In[8]:


# 산업별로 종목 수는 몇 개인가?
import pandas as pd
import numpy as np
stock = pd.read_csv('World-Stock-Prices-Dataset.csv', low_memory = False)
stock['Date'] = list(stock['Date'].str.slice(0,10))


# In[9]:


industries = np.unique(stock['Industry_Tag'])
for i in industries:    
    stockrows = stock[stock['Industry_Tag'] == i].duplicated('Ticker', keep = 'last')
    print(i,sum(~stockrows))


# In[10]:


# hospitality 산업에 있는 종목을 찾아보자
select = stock['Industry_Tag'] == 'hospitality'
hospi = stock[select]
tickers = np.unique(hospi['Brand_Name'])
tickers


# In[19]:


# 각 종목의 수익률 구하기 - 딕셔너리로
hospi_profit = {}
for t in tickers:
    hospi_profit[t] = hospi.loc[(hospi['Date'] == '2023-09-18') & (hospi['Brand_Name'] == t)]['Close'].values / hospi.loc[(hospi['Date'] == '2023-03-20') & (hospi['Brand_Name'] == t)]['Close'].values * 100 - 100
hospi_profit


# In[20]:


#최저, 최고수익률 종목 찾기
for k, v in hospi_profit.items():
    if v == min(hospi_profit.values()):
        hospi_min = k
for k, v in hospi_profit.items():
    if v == max(hospi_profit.values()):
        hospi_max = k
        
print('최저수익률 =', hospi_min, ', 최고수익률 =', hospi_max)


# In[33]:


# 시각화
import matplotlib.pyplot as plt
x_val = hospi.loc[(hospi['Brand_Name'] == hospi_min)]['Date']
y_val = hospi.loc[(hospi['Brand_Name'] == hospi_min)]['Close']
x_val2 = hospi.loc[(hospi['Brand_Name'] == hospi_max)]['Date']
y_val2 = hospi.loc[(hospi['Brand_Name'] == hospi_max)]['Close']


# In[35]:


fig, axs = plt.subplots(1, 2, figsize = (10, 4))

axs[0].plot(x_val, y_val)
axs[0].set_title('Hilton')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Stock Price')

axs[1].plot(x_val2, y_val2)
axs[1].set_title('Marriott')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Stock Price')

fig.show()

