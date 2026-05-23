import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.title('Movie recomender')
st.write('Find the perfect movie, that suits your mood😁')


df=pd.read_csv('tmdb_5000_movies.csv')


movies=df[['title','overview','genres','keywords','revenue']]
movies.dropna(inplace=True)
movies['tags']=movies['overview']+movies['genres']+movies['keywords']

cv=CountVectorizer(max_features=5000,stop_words='english')

vectors=cv.fit_transform(movies['tags']).toarray()


similarity=cosine_similarity(vectors)

movie_list=movies['title'].values

selected_movie=st.selectbox('select_movie',movie_list)

def recomended(movie):
    index=movies[movies['title']==movie].index[0]
    distance=similarity[index]
    movies_list=sorted(list(enumerate(distance)),
                       reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_list:
        st.write(movies.iloc[i[0]].title)
        
if st.button('check'):
    recomended(selected_movie)
