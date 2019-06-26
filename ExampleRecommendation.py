#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt # data visualization library


# In[126]:


df = pd.read_csv('../ml-20m/rating.csv',sep=',', names=['user_id','movieId','rating','titmestamp'])
df.head()

movie_titles = pd.read_csv('../ml-20m/movies.csv')
movie_titles.head()


# In[127]:


df = pd.merge(df, movie_titles, on='movieId')
df.rating = df.rating.astype(int)
df.head()


# In[128]:


df.describe()


# In[129]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()


# In[130]:


ratings['number_of_ratings'] = df.groupby('title')['rating'].count()
ratings.head()


# In[131]:


ratings['rating'].hist(bins=50)


# In[132]:


ratings['number_of_ratings'].hist(bins=60)


# In[133]:


sns.jointplot(x='rating', y='number_of_ratings', data=ratings)


# In[142]:


movie_matrix = df.pivot_table(index='user_id', columns='title', values='rating')
movie_matrix.head()


# In[143]:


ratings.sort_values('number_of_ratings', ascending=False).head(10)


# In[145]:


AFO_user_rating = movie_matrix['Pulp Fiction (1994)']
contact_user_rating = movie_matrix['Shawshank Redemption, The (1994)']


# In[146]:


AFO_user_rating.head()
contact_user_rating.head()


# In[147]:


similar_to_air_force_one=movie_matrix.corrwith(AFO_user_rating)


# In[148]:


similar_to_air_force_one.head()


# In[149]:


similar_to_contact = movie_matrix.corrwith(contact_user_rating)


# In[150]:


similar_to_contact.head()


# In[155]:


corr_contact = pd.DataFrame(similar_to_contact, columns=['Correlation'])
corr_contact.dropna(inplace=True)
corr_contact.head(50)


# In[154]:



corr_AFO = pd.DataFrame(similar_to_air_force_one, columns=['correlation'])
corr_AFO.dropna(inplace=True)
corr_AFO.head(50)


# In[156]:


corr_AFO = corr_AFO.join(ratings['number_of_ratings'])
corr_contact = corr_contact.join(ratings['number_of_ratings'])
corr_AFO .head()
corr_contact.head()


# In[157]:


corr_AFO[corr_AFO['number_of_ratings'] > 100].sort_values(by='correlation', ascending=False).head(10)


# In[158]:


corr_contact[corr_contact['number_of_ratings'] > 100].sort_values(by='Correlation', ascending=False).head(10)


# In[71]:


similar_to_air_force_one=movie_matrix.corrwith(AFO_user_rating)


# In[72]:


similar_to_air_force_one.head()


# In[ ]:




