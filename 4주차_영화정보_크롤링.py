#!/usr/bin/env python
# coding: utf-8

# ### 셀레니움을 활용한 영화 기본정보 제공 프로그램 제작
# #### 네이버 시리즈온에서 top 100 영화들 기본정보 가져오기

# In[1]:


import pandas as pd
import numpy as np
from selenium import webdriver
import chromedriver_autoinstaller
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By


# In[2]:


def get_info3():
    import pandas as pd
    import numpy as np
    from selenium import webdriver
    import chromedriver_autoinstaller
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By
    chrome_options = webdriver.ChromeOptions()
    #chrome_options.binary_location = r'C:\Program Files\Google\Chrome\Application\chrome.exe'  # 본인의 크롬 브라우저 설치 경로를 찾아 넣어주세요.
    chrome_options.add_argument('--headless')  # 브라우저 창을 열지 않고 headless 모드로 실행하는 옵션입니다. 생략하셔도 무방합니다.
    driver = webdriver.Chrome(options=chrome_options)
    #크롬을 불러오는 방법들 설정들 완료
    
    
    driver.get('https://serieson.naver.com/v3/movie/ranking/monthly')
    choice = input('너가 알고싶은 영화는? ')
    film_name = driver.find_elements(By.CLASS_NAME, 'Title_title__s9o0D')
    for name in film_name:
        if name.text == choice:
            parent_href = name.find_element(By.XPATH, '../..').get_attribute('href')
            break

    driver.get(parent_href)
    rate = driver.find_element(By.XPATH,'/html/body/div/div[2]/div[1]/div[2]/div/div[1]/div[2]/span[1]')
    release_year = driver.find_element(By.XPATH, '/html/body/div/div[2]/div[1]/div[2]/div/div[1]/div[2]/span[2]')
    run_time = driver.find_element(By.XPATH, '/html/body/div/div[2]/div[1]/div[2]/div/div[1]/div[2]/span[3]')
    age = driver.find_element(By.XPATH, '/html/body/div/div[2]/div[1]/div[2]/div/div[1]/div[2]/span[4]')
    print('평점:', rate.text)
    print('개봉년도:',release_year.text)
    print('러닝타임:',run_time.text)
    print('연령제한:',age.text)


# In[3]:


def get_info():
    import pandas as pd
    import numpy as np
    from selenium import webdriver
    import chromedriver_autoinstaller
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By    
    import pandas as pd
    import numpy as np
    df = pd.read_csv('top100.csv', encoding = 'CP949')
    df = df.drop([43, 46, 52, 77]).reset_index(drop=True)
    chrome_options = webdriver.ChromeOptions()
    chrome_options.binary_location = r'C:\Program Files\Google\Chrome\Application\chrome.exe'  # 본인의 크롬 브라우저 설치 경로를 찾아 넣어주세요.
    chrome_options.add_argument('--headless')  # 브라우저 창을 열지 않고 headless 모드로 실행하는 옵션입니다. 생략하셔도 무방합니다.
    driver = webdriver.Chrome(options=chrome_options)

    for href in df['Href']:
        driver.get('https://serieson.naver.com' + href)
        k = driver.find_element(By.CSS_SELECTOR,'div.info')
        print(k.text)


# In[4]:


def get_info2():
    import pandas as pd
    import numpy as np
    from selenium import webdriver
    import chromedriver_autoinstaller
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By
    chrome_options = webdriver.ChromeOptions()
    #chrome_options.binary_location = r'C:\Program Files\Google\Chrome\Application\chrome.exe'  # 본인의 크롬 브라우저 설치 경로를 찾아 넣어주세요.
    chrome_options.add_argument('--headless')  # 브라우저 창을 열지 않고 headless 모드로 실행하는 옵션입니다. 생략하셔도 무방합니다.
    driver = webdriver.Chrome(options=chrome_options)
    #크롬을 불러오는 방법들 설정들 완료
    
    
    driver.get('https://serieson.naver.com/v3/movie/ranking/monthly')
    choice = input('너가 알고싶은 영화는? ')
    film_name = driver.find_elements(By.CLASS_NAME, 'Title_title__s9o0D')
    for name in film_name:
        if name.text == choice:
            parent_href = name.find_element(By.XPATH, '../..').get_attribute('href')
            break

    driver.get(parent_href)
    info = driver.find_element(By.CSS_SELECTOR,'div.info')
    min = info.text.find('분')
    space = info.text.find(' ')
    review = info.text[0:4]
    year = info.text[4:8]
    run_time = info.text[8:min+1]
    age = info.text[min + 1: space + 4]
    print('평점: ' + review+',', '개봉년도: ' + year+',', '러닝타임: '+run_time+',','관람등급: ' + age)


# In[5]:


def show_films():
    import pandas as pd
    import numpy as np
    from selenium import webdriver
    import chromedriver_autoinstaller
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By
    chrome_options = webdriver.ChromeOptions()
    #chrome_options.binary_location = r'C:\Program Files\Google\Chrome\Application\chrome.exe'  # 본인의 크롬 브라우저 설치 경로를 찾아 넣어주세요.
    chrome_options.add_argument('--headless')  # 브라우저 창을 열지 않고 headless 모드로 실행하는 옵션입니다. 생략하셔도 무방합니다.
    driver = webdriver.Chrome(options=chrome_options)
    driver.get('https://serieson.naver.com/v3/movie/ranking/monthly')
    titles = driver.find_elements(By.CLASS_NAME,'Title_title__s9o0D')
    film_lst = []
    for title in titles:
        film_lst.append(title.text)
    print(film_lst)


# ### 정규세션 과제
# ##### 박재완, 허승욱

# In[ ]:


import pandas as pd
import numpy as np
from selenium import webdriver
import chromedriver_autoinstaller
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
chrome_options = webdriver.ChromeOptions()
chrome_options.binary_location = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
chrome_options.add_argument('--headless')
driver = webdriver.Chrome(options=chrome_options)
df = pd.read_csv('top100.csv', encoding = 'CP949')
df = df.drop([43, 46, 52, 77]).reset_index(drop=True)




# In[34]:


rate_list = []
year_list = []
runtime_list = []
age_list = []
like_list = []
price_list = []

for film in df['Href']:
    driver.get('https://serieson.naver.com' + film)
    #평점, 년도 등 필요한 html 경로를 XPATH로 변수에 초기화.
    rate = driver.find_element(By.XPATH,'/html/body/div/div[2]/div[1]/div[2]/div/div[1]/div[2]/span[1]')
    year = driver.find_element(By.XPATH,'/html/body/div/div[2]/div[1]/div[2]/div/div[1]/div[2]/span[2]')
    runtime = driver.find_element(By.XPATH,'/html/body/div/div[2]/div[1]/div[2]/div/div[1]/div[2]/span[3]')
    age = driver.find_element(By.XPATH,'/html/body/div/div[2]/div[1]/div[2]/div/div[1]/div[2]/span[4]')
    like = driver.find_element(By.XPATH,'/html/body/div/div[2]/div[1]/div[2]/div/ul/li[2]/div/a/em[2]')
    price = driver.find_element(By.XPATH,'/html/body/div/div[2]/div[1]/div[2]/div/div[2]/button/span')
    rate_list.append(rate.text)
    year_list.append(year.text)
    runtime_list.append(runtime.text.replace('분',''))
    if age.text == '12세 관람가':
        age_list.append(12)
    elif age.text == '15세 관람가':
        age_list.append(15)
    else:
        age_list.append(0)
    like_list.append(like.text.replace(',',''))
    price_list.append(price.text.replace(',',''))


# In[ ]:


#df에 새로운 칼럼들 추가.
df['Rate'] = rate_list
df['Year'] = year_list
df['Runtime'] = runtime_list
df['Age'] = age_list
df['Like'] = like_list
df['Price'] = price_list
#영화 콩나물 삭제.
df.drop(labels = 79, axis = 0, inplace = True)

#각 칼럼을 숫자데이터로 변환.
df['Rate'] = df['Rate'].astype('float')
df['Year'] = df['Year'].astype('int')
df['Runtime'] = df['Runtime'].astype('int')
df['Like'] = df['Like'].astype('int')
df['Price'] = df['Price'].astype('int')


# In[42]:


print(df.info())
df.head()


# ### 칼럼별 순위 산점도

# In[82]:


import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['font.size'] = 7.5
fig, ax = plt.subplots(2,3)
plt.rcParams["figure.figsize"] = (10,5)

ax[0][0].scatter(df['Ranking'],df['Rate'], s=15,  edgecolors="black", alpha=0.3)
ax[0][1].scatter(df['Ranking'],df['Year'], s=15,  edgecolors="black", alpha=0.3)
ax[0][2].scatter(df['Ranking'],df['Runtime'], s=15,  edgecolors="black", alpha=0.3)
ax[1][0].scatter(df['Ranking'],df['Age'], s=15,  edgecolors="black", alpha=0.3)
ax[1][1].scatter(df['Ranking'],df['Like'], s=15,  edgecolors="black", alpha=0.3)
ax[1][2].scatter(df['Ranking'],df['Price'], s=15,  edgecolors="black", alpha=0.3)
ax[0][0].set_title('평점에 따른 순위')
ax[0][1].set_title('개봉년도에 따른 순위')
ax[0][2].set_title('상영시간에 따른 점수')
ax[1][0].set_title('상영등급에 따른 점수')
ax[1][1].set_title('관심수에 따른 점수')
ax[1][2].set_title('가격에 따른 점수')
plt.show()


# ### 칼럼별 순위와의 상관계수

# In[79]:


corr = df.corr()
print(corr)
sns.heatmap(corr, cmap = 'viridis')


# In[46]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 데이터에서 Runtime, Ranking 열을 선택
Runtime = df[['Runtime']].values
Ranking = df[['Ranking']].values


# 데이터를 훈련 세트와 테스트 세트로 분할
Runtime_train, Runtime_test, Ranking_train, Ranking_test = train_test_split(Runtime, Ranking, test_size=0.2, random_state=42)


# LinearRegression 모델을 생성
model2 = LinearRegression()

# 훈련셋을 훈련시킴
model2.fit(Runtime_train, Ranking_train)
# 훈련셋의 스코어 확인
print(model2.score(Runtime_train, Ranking_train))
print(model2.score(Runtime_test, Ranking_test))

