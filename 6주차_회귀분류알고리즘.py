#!/usr/bin/env python
# coding: utf-8

# ### 실습데이터: 와인 분류
# - 모든 특성은 수치형 데이터
# - 타겟은 quality
# 1. 아래 모델을 사용하여 모델의 성능을 비교(매개변수는 자유롭게)
# - LogisticRegression
# - SGDClassifier
# - RandomforestClassifier
# 2. 그리드 서치로 각 모델의 매개변수 탐색 후 재비교
# - 각 모델의 매개변수들을 직접 확인하면서 가능한 값 리스트를 생성

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv('wine_classification.csv')
df.head()


# In[5]:


df.corr()


# In[6]:


df.info()


# In[7]:


df.describe()


# ## 1. 랜덤 포레스트

# In[8]:


# Id, Quality 제외
selected_columns = df.drop(columns=['Id','quality']).columns
selected_columns


# In[9]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data=df.drop(columns=['Id','quality']).to_numpy()  # id 컬럼 제외
target=df['quality'].to_numpy()

train_input,test_input,train_target,test_target=train_test_split(data,target,test_size=0.2,random_state=42)


# In[10]:


from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_jobs=-1,random_state=42)
scores=cross_validate(rf,train_input,train_target,
                     return_train_score=True,n_jobs=-1)
print(np.mean(scores['train_score']),np.mean(scores['test_score']))


# 훈련세트에 과대적합

# #### 하이퍼 파라미터 튜닝

# In[11]:


from sklearn.model_selection import GridSearchCV

params = { 'n_estimators' : [10, 100],
           'max_depth' : [6, 8, 10, 12],
           'min_samples_leaf' : [8, 12, 18],
           'min_samples_split' : [8, 16, 20]
            }

# RandomForestClassifier 객체 생성 후 GridSearchCV 수행
rf_clf = RandomForestClassifier(random_state = 0, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid = params, cv = 3, n_jobs = -1)
grid_cv.fit(train_input,train_target)

print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))


# ##### 상관관계 하위 3개 삭제시-> 정확도 소폭 하락

# In[12]:


data_1=df.drop(columns=['Id','quality','residual sugar','pH','chlorides']).to_numpy()  # id 컬럼 제외
target_1=df['quality'].to_numpy()

train_input,test_input,train_target,test_target=train_test_split(data_1,target_1,test_size=0.2,random_state=42)

params = { 'n_estimators' : [10, 100],
           'max_depth' : [6, 8, 10, 12],
           'min_samples_leaf' : [8, 12, 18],
           'min_samples_split' : [8, 16, 20]
            }

# RandomForestClassifier 객체 생성 후 GridSearchCV 수행
rf_clf = RandomForestClassifier(random_state = 0, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid = params, cv = 3, n_jobs = -1)
grid_cv.fit(train_input,train_target)

print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))


# ## HGB 사용

# In[13]:


from sklearn.ensemble import HistGradientBoostingClassifier

hgb=HistGradientBoostingClassifier(random_state=42)
scores=cross_validate(hgb,train_input,train_target,
                     return_train_score=True)
print(np.mean(scores['train_score']),np.mean(scores['test_score']))


# 여전히 과대적합

# ### 특성의 중요도

# In[26]:


from sklearn.inspection import permutation_importance

result=permutation_importance(hgb,train_input,train_target,
                             n_repeats=10,random_state=42,n_jobs=-1)


# In[28]:


importances_mean = result.importances_mean
feature_names = df.drop(columns=['Id','quality']).columns  
# 중요도와 이름을 묶어서 출력
feature_importances = list(zip(feature_names, importances_mean))

# 중요도가 높은 순으로 정렬
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

# 출력
for feature, importance in feature_importances:
    print(f"{feature}: {importance}")


# ### 1) citric acid 삭제

# In[29]:


data2=df.drop(columns=['Id','quality','citric acid']).to_numpy()  
target2=df['quality'].to_numpy()

train_input,test_input,train_target,test_target=train_test_split(data2,target2,test_size=0.2,random_state=42)


# In[30]:


hgb=HistGradientBoostingClassifier(random_state=42)
scores=cross_validate(hgb,train_input,train_target,
                     return_train_score=True)
print(np.mean(scores['train_score']),np.mean(scores['test_score']))


# 오히려 성능 소폭감소

# ### 2) volatile acidity,fixed acidity,residual sugar,citric acid

# In[31]:


data3=df.drop(columns=['Id','quality','citric acid','fixed acidity','volatile acidity','residual sugar']).to_numpy()  
target3=df['quality'].to_numpy()

train_input,test_input,train_target,test_target=train_test_split(data3,target3,test_size=0.2,random_state=42)


# In[32]:


hgb=HistGradientBoostingClassifier(random_state=42)
scores=cross_validate(hgb,train_input,train_target,
                     return_train_score=True)
print(np.mean(scores['train_score']),np.mean(scores['test_score']))


# 성능이 추가로 감소

# # 2.SGD

# In[34]:


#입력 데이터, 타깃 데이터 나누기

wine_input = df[['fixed acidity','volatile acidity','citric acid','residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].to_numpy()
wine_target = df['quality'].to_numpy()

#훈련 세트와 테스트 세트로 나누기
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(wine_input, wine_target, random_state=42)

#표준화 전처리
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

#훈련 세트에서 학습한 통계 값으로 테스트 세트도 변환해야 함
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


# In[35]:


#확률적 경사 하강법을 제공하는 대표적인 분류용 클래스
from sklearn.linear_model import SGDClassifier

#loss는 손실 함수의 종류를 지정, max_iter은 에포크 횟수 지정 (전체 훈련 세트 10회 반복)
sc = SGDClassifier(loss = 'log_loss', max_iter = 10, random_state = 42)
sc.fit(train_scaled, train_target)

#정확도 점수 출력
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))


# 훈련 세트와 테스트 세트 정확도가 낮은 걸로 보아 지정된 반복 횟수가 부족하다 partial_fit(): 모델을 이어서 추가로 훈련

# In[36]:


sc.partial_fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))


# 오히려 정확도가 감소

# In[37]:


#에포크와 과대/과소적합
#에포크 횟수가 적으면 모델이 훈련 세트를 덜 학습하고 - 과소적합
#에포크 횟수가 충분하면 훈련 세트를 완전히 학습한다 - 너무 많으면 과대적합
#테스트 세트 점수가 감소하기 시작하는 순간이 과대적합되기 시작하는 곳
import numpy as np

sc = SGDClassifier(loss = 'log_loss', random_state = 42)

#에포크마다 훈련 세트와 테스트 세트에 대한 점수를 기록한다
train_score = []
test_score = []

#partial_fit() 메서드만 사용하려면 훈련 세트에 있는 전체 클래스의 레이블을 전달해주어야 한다
#7개 목록을 만든다
classes = np.unique(train_target)


# In[38]:


#300번의 에포크
#반복마다 훈련 세트와 테스트 세트의 점수를 계산하여 리스트에 추가
for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes = classes)

    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))


# In[39]:


#훈련 세트와 테스트 세트의 점수를 그래프로 그려보기
import matplotlib.pyplot as plt

plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()    


# 파란색이 훈련 세트 그래프, 주황색이 테스트 세트 그래프  
# 
# 초기에는 과소적합, 200번째 에포크 이후에는 훈련세트와 테스트 세트의 점수가 벌어지고 있다

# In[40]:


#반복 횟수 100으로 모델 다시 훈련, 훈련 세트와 테스트 세트에서 점수 출력
sc = SGDClassifier(loss = 'log_loss', max_iter = 200, tol = None, random_state = 42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))


# In[41]:


#loss 매개변수의 기본값은 hinge다.
#힌지 손실을 사용해 같은 반복 횟수 동안 모델을 훈련
sc = SGDClassifier(loss = 'hinge', max_iter = 200, tol = None, random_state = 42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))


# ## 3.로지스틱 회귀

# In[5]:


input_data = df[['fixed acidity', 'volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
target_data = df['quality']


# In[6]:


from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(input_data, target_data,test_size = 0.2,  random_state = 42)


# In[7]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_input, train_target)


# In[8]:


print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))


# ### 다중공선성 확인

# In[9]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 상수항 추가
X = sm.add_constant(df)

# VIF 계산
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# 결과 출력
print(vif_data)


# 기준은 10. 문제되는 특성은 없음

# ### 특성 중요도 확인
# - 랜덤포레스트 특성 중요도
# - 순열 중요도
# - GBC 특성 중요도
# - LGBMC 특성 중요도

# In[10]:


import matplotlib.pyplot as plt


# 

# In[11]:


#랜덤포레스트 특성 중요도
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd

# 랜덤 포레스트 모델 생성
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(train_input, train_target)

# 특성 중요도 얻기
rf_feature_importances = rf_model.feature_importances_
plt.bar(input_data.columns, rf_feature_importances)


# sulphates, alcohol이 높음

# In[12]:


#순열 중요도
from sklearn.inspection import permutation_importance

# 랜덤 포레스트 모델 생성
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(train_input, train_target)

# 특성 중요도 얻기
piresult = permutation_importance(rf_model, test_input, test_target, n_repeats=30, random_state=42)

plt.bar(input_data.columns, piresult.importances_mean)


# 마찬가지로 sulphates와 alcohol이 높음

# In[13]:


from sklearn.ensemble import GradientBoostingClassifier

# 모델 생성 및 학습
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(train_input, train_target)

# 특성 중요도 확인
gbfeature_importances = gb.feature_importances_
plt.bar(input_data.columns, gbfeature_importances)


# 마찬가지로 sulphates와 alcohol

# In[14]:


import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgb_model.fit(train_input, train_target)

plt.bar(input_data.columns, lgb_model.feature_importances_)


# 특정 특성이 우세하다고 보기 어려움

# 따라서 sulphates와 alcohol을 특성으로 다시 로지스틱회귀 돌려보자.

# In[16]:


mod_input_data = df[['sulphates','alcohol']]
mod_target_data = df[['quality']]


# In[17]:


from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(mod_input_data, mod_target_data,test_size = 0.2,  random_state = 42)


# In[18]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_input, train_target)
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))


# 과적합 개선, 점수도 56, 51 -> 57, 56 으로 개선

# ### 표준화
# 표준화를 안 한 경우의 점수가 더 높음  
# 정말 미세한 차이이지만 MinMaxScaler가 가장 높음

# In[19]:


from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
mm.fit(train_input)
train_scaled = mm.transform(train_input)
test_scaled = mm.transform(test_input)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))


# In[20]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))


# In[21]:


from sklearn.preprocessing import RobustScaler
rb = RobustScaler()
rb.fit(train_input)
train_scaled = rb.transform(train_input)
test_scaled = rb.transform(test_input)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))


# ### Polynomial 특성변환
# 0.006 상승.. 57.1/56 -> 57.7/55.5 

# In[22]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2, include_bias = False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)
print(poly.get_feature_names_out())


# In[23]:


lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))


# ### 그리드서치
# 과적합 개선 57.7/55.5 -> 57.2/56

# In[24]:


from sklearn.model_selection import GridSearchCV
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1.0, 10.0, 20],
    'solver': [ 'lbfgs', 'liblinear', 'newton-cg' ],
    'max_iter': [100, 200]
}
grid_search = GridSearchCV(lr, param_grid, cv=5)
grid_search.fit(train_input, train_target)
print(grid_search.score(train_input, train_target))
print(grid_search.score(test_input, test_target))
print(grid_search.best_params_)


# ### 후진제거법 후 재학습
# 하나씩 학습하면서 필요없는 특성들을 제거.  
# 아래 코드에서는 4개까지 줄임. 오히려 성능저하  
# 특성은 마지막 4개가 추출됨.  
# 
# 57.2/56 -> 56.7/53.8

# In[25]:


from sklearn.model_selection import train_test_split
wtrain_input, wtest_input, wtrain_target, wtest_target = train_test_split(input_data, target_data,test_size = 0.2,  random_state = 42)


# In[26]:


# 반복적 변수 제거를 하여 변수별 중요도를 도출
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = 20, max_iter = 100, penalty = 'l2', solver = 'lbfgs')

# 모델 입력, n_features_to_select: 골라낼 변수의 수, step: 한번에 몇개씩 제거할지 선택
rfe = RFE(lr, n_features_to_select=4, step=1)
model = rfe.fit(wtrain_input,wtrain_target)
# 선택될 변수
model.support_

# 변수 중요도(숫자 높을수록 불필요하다)
print(model.score(wtrain_input,wtrain_target))
print(model.score(wtest_input,wtest_target))
print(model.support_)


# 최종적으로 가장 높은 점수는 57/56.

# ### 결론

# 세 가지 모델을 비교했을 때, 사용한 특성과 매개변수들이 모두 다르지만, 랜덤포레스트, SGD = 로지스틱 순으로 성능이 평가됨.
