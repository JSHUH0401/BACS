#!/usr/bin/env python
# coding: utf-8

# # **월간 데이콘 신용카드 사용자 연체 예측 AI 경진대회**
# **알고리즘 | 정형 | 분류 | 금융 | LogLoss** 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('train_dataset.csv')
df


# In[3]:


df.describe()


# ## 범주형 데이터에 대한 시각화

# credit별 개수

# In[6]:


sns.countplot(x = 'credit', data = df)


# In[9]:


object_cols = df[['child_num','income_total','DAYS_BIRTH','DAYS_EMPLOYED','family_size','begin_month']].columns


# In[10]:


fig, axs = plt.subplots(3, 2, figsize=(22, 15))

for i in range(3):
    for j in range(2):
        idx = j * 3 + i
        if idx < len(object_cols):
            sns.scatterplot(data=df, x="credit", y=object_cols[idx], ax=axs[i, j])
            axs[i, j].set_title(f'{object_cols[idx]} by credit')
            axs[i, j].grid(True)


# In[4]:


def new_df():
    ndf = pd.read_csv('train_dataset.csv')
    return ndf


# In[33]:


def xgboost(x,y):
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV

    # 훈련 세트와 테스트 세트로 분할
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    param_grid = {
        'learning_rate': [0.1, 0.2, 0.3],
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7]
    }

    grid_search = GridSearchCV(xgb.XGBClassifier(objective="multi:softmax", n_estimators = 200, num_class=3), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print("최적의 매개변수:", grid_search.best_params_)

    # 테스트 세트에서 성능 평가
    accuracy = grid_search.best_estimator_.score(X_test, y_test)
    print(f"테스트 세트 정확도: {accuracy:.4f}")


# In[12]:


def jxgboost(x,y):
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV

    # 훈련 세트와 테스트 세트로 분할
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    print(model.score(X_train, y_train))
    print(model.score(X_test, y_test))
    test = pd.read_csv('test.csv')
    
    return model.predict_proba()


# In[92]:


def preprocessing(df):
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import LabelEncoder # 레이블 인코더

    label_encoder = LabelEncoder()
    rb = RobustScaler()
    
    # 레이블 인코딩 적용
    df['gender'] = label_encoder.fit_transform(df['gender'])
    df['car'] = label_encoder.fit_transform(df['car'])
    df['reality'] = label_encoder.fit_transform(df['reality'])
    df['income_type'] = label_encoder.fit_transform(df['income_type'])
    df['edu_type'] = label_encoder.fit_transform(df['edu_type'])
    df['family_type'] = label_encoder.fit_transform(df['family_type'])
    df['house_type'] = label_encoder.fit_transform(df['house_type'])
    df['DAYS_BIRTH'] = df['DAYS_BIRTH'].abs()
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].abs()
    df['begin_month'] = df['begin_month'].abs()
    df.drop(['index','FLAG_MOBIL'],axis = 1, inplace = True)
    employ = df['DAYS_EMPLOYED'].to_numpy().reshape(-1,1)
    rb.fit(employ)
    df['DAYS_EMPLOYED'] = rb.transform(employ)
    
    return df


# # LogLoss 기준 우리 조의 최고 모델

# In[93]:


nndf = new_df()
nndf = preprocessing(nndf)
x = nndf.drop(['credit'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# ### 이외에 우리가 한 것들

# 로그함수로 고용일자 변환

# In[94]:


df = new_df()
log10_data = np.log10(df['DAYS_EMPLOYED'])
log10data = input_data
log10data['DAYS_EMPLOYED'] = log10_data
log10data


# In[95]:


log_data = np.log(df['DAYS_EMPLOYED'])
logdata = input_data
logdata['DAYS_EMPLOYED'] = log_data
logdata


# In[ ]:


xgboost(log10_data, target_data)
xgboost(log_data, target_data)


# ## 전체에서 하나씩 빼보면서 영향을 크게 주는 특성 추출

# In[ ]:


nndf = new_df()
x = nndf.drop(['credit','index','email'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# In[ ]:


nndf = new_df()
x = nndf.drop(['credit','index','gender'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# In[ ]:


nndf = new_df()
x = nndf.drop(['credit','index','car'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# In[ ]:


nndf = new_df()
x = nndf.drop(['credit','index','reality'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# In[ ]:


nndf = new_df()
x = nndf.drop(['credit','index','child_num'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# In[ ]:


nndf = new_df()
x = nndf.drop(['credit','index','income_total'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# In[ ]:


nndf = new_df()
x = nndf.drop(['credit','index','income_type'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# In[ ]:


nndf = new_df()
x = nndf.drop(['credit','index','edu_type'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# In[ ]:


nndf = new_df()
x = nndf.drop(['credit','index','family_type'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# In[ ]:


nndf = new_df()
x = nndf.drop(['credit','index','house_type'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# In[ ]:


nndf = new_df()
x = nndf.drop(['credit','index','DAYS_BIRTH'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# In[ ]:


nndf = new_df()
x = nndf.drop(['credit','index','DAYS_EMPLOYED'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# In[ ]:


nndf = new_df()
x = nndf.drop(['credit','index','family_size'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# In[ ]:


nndf = new_df()
x = nndf.drop(['credit','index','phone'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# In[ ]:


nndf = new_df()
x = nndf.drop(['credit','index','work_phone'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# In[ ]:


nndf = new_df()
x = nndf.drop(['credit','index','begin_month'], axis = 1)
y = nndf['credit']
xgboost(x,y)


# ## 위 특성 중 상위 4개를 ploynomial 특성공학으로 추가   ['income_total','DAYS_EMPLOYED','email','index']

# In[ ]:


from sklearn.model_selection import train_test_split

ndf = new_df()
ext = df[['income_total','DAYS_EMPLOYED']]
ext_target = df['credit']

X_train, X_test, y_train, y_test = train_test_split(ext, ext_target, test_size=0.2, random_state=42)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2, include_bias = False)
poly.fit(X_train)
train_poly = poly.transform(X_train)
test_poly = poly.transform(X_test)
train_poly = pd.DataFrame(train_poly)
df_combined = pd.concat([df.drop(['income_total','DAYS_EMPLOYED','email','index'],axis = 1),train_poly], axis = 1)


# In[ ]:


df_combined = pd.concat([df.drop(['income_total','DAYS_EMPLOYED','email','index'],axis = 1),train_poly], axis = 1)


# In[17]:


test = pd.read_csv('test.csv')
ptest = preprocessing(test)
ptest.head()


# ## CatBoost

# In[49]:


import catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler

train = new_df()
ind = train.drop(['index','credit','FLAG_MOBIL'],axis = 1)
tard = train['credit']
rb = RobustScaler()
employ = train['DAYS_EMPLOYED'].to_numpy().reshape(-1,1)
rb.fit(employ)
train['DAYS_EMPLOYED'] = rb.transform(employ)
# 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(ind, tard, test_size=0.2, random_state=42)

model = CatBoostClassifier(loss_function = 'MultiClass', iterations = 200)
param_grid = {

    'learning_rate': [0.1, 0.3, 0.5],
    'depth': [3, 6,9]
}
grid = GridSearchCV(model, param_grid, cv = 5, scoring = 'accuracy')
grid.fit(X_train, y_train)
print(grid.best_params_)

print(grid.score(X_train, y_train))
print(grid.score(X_test, y_test))


# In[52]:


import catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler

train = new_df()
ind = train.drop(['index','credit','FLAG_MOBIL'],axis = 1)
tard = train['credit']
rb = RobustScaler()
employ = ind['DAYS_EMPLOYED'].to_numpy().reshape(-1,1)
rb.fit(employ)
ind['DAYS_EMPLOYED'] = rb.transform(employ)

n = train[['income_total','DAYS_BIRTH','family_size','begin_month']]
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2, include_bias = False)
poly.fit(n)
train_poly = poly.transform(n)
train_poly = pd.DataFrame(train_poly)
df_combined = pd.concat([ind.drop(['income_total','DAYS_BIRTH','family_size','begin_month'],axis = 1),train_poly], axis = 1)
# 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(df_combined, tard, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(objective = 'multi:softmax', n_estimators = 200, num_class =3, learning_rate = 0.2, max_depth = 7)
#param_grid = {
#    'learning_rate': [0.1, 0.3, 0.5],
#    'depth': [3, 6,9]
#}
#grid = GridSearchCV(model, param_grid, cv = 5, scoring = 'accuracy')
model.fit(X_train, y_train)
#print(grid.best_params_)

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
y_proba = model.predict_proba(X_test)

# Log Loss 출력
logloss = log_loss(y_test, y_proba)
print(f'Log Loss: {logloss}')


# In[80]:


rb = RobustScaler()
temploy = ptest['DAYS_EMPLOYED'].to_numpy().reshape(-1,1)
rb.fit(temploy)
ptest['DAYS_EMPLOYED'] = rb.transform(temploy)
poly = PolynomialFeatures(degree = 2, include_bias = False)
k = ptest[['income_total','DAYS_BIRTH','family_size','begin_month']]

poly.fit(k)
test_poly = poly.transform(k)
test_poly = pd.DataFrame(test_poly)
test_combined = pd.concat([ptest.drop(['income_total','DAYS_BIRTH','family_size','begin_month'],axis = 1),test_poly], axis = 1)
test_combined.head()


# In[82]:


submit = model.predict_proba(test_combined)


# In[83]:


sub = pd.read_csv('sample_submission.csv')
sub.loc[:,1:]=submit
sub
sub.to_csv('submit.csv')


# In[ ]:


sub.to_csv('submit.csv')


# ### 최종 점수는 65점
