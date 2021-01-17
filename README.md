# 영화 추천 인공지능 만들기 ([유튜브 빵형의 개발도상국](https://www.youtube.com/watch?v=mLwMe4KUZz8))

 
### 프로젝트의 주요 기능
- 이것은 영화 추천 프로젝트입니다.

![ScreenCaptureProject11](https://user-images.githubusercontent.com/62390565/104807246-e8a11280-5820-11eb-91d9-a37a83f0bbf8.gif)

### 데이터 셋
- The Movie Dataset : https://www.kaggle.com/rounakbanik/the-movies-dataset

### 블로그
- [공부한 내용 블로그에 작성](https://rkaclfrns.tistory.com/8)


### 코드
```python3
meta = pd.read_csv('C:/Users/공부용/movie_recommendation_engine/the-movies-dataset/movies_metadata.csv')

meta = meta[['id', 'original_title', 'original_language', 'genres']]
meta = meta.rename(columns={'id':'movieId'})
meta = meta[meta['original_language'] == 'en']


ratings = pd.read_csv('C:/Users/공부용/movie_recommendation_engine/the-movies-dataset/ratings_small.csv')
ratings = ratings[['userId', 'movieId', 'rating']]

meta.movieId = pd.to_numeric(meta.movieId, errors='coerce')
ratings.movieId = pd.to_numeric(ratings.movieId, errors='coerce')
data = pd.merge(ratings, meta, on='movieId', how='inner')


matrix = data.pivot_table(index='userId', columns='original_title', values='rating')


GENRE_WEIGHT = 0.1

def pearsonR(s1, s2):
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    return np.sum(s1_c * s2_c) / np.sqrt(np.sum(s1_c ** 2) * np.sum(s2_c ** 2))

def recommend(input_movie, matrix, n, similar_genre=True):
    input_genres = meta[meta['original_title'] == input_movie]['genres'].iloc(0)[0]

    result = []
    for title in matrix.columns:
        if title == input_movie:
            continue

        # rating comparison
        cor = pearsonR(matrix[input_movie], matrix[title])
        
        # genre comparison
        if similar_genre and len(input_genres) > 0:
            temp_genres = meta[meta['original_title'] == title]['genres'].iloc(0)[0]

            same_count = np.sum(np.isin(input_genres, temp_genres))
            cor += (GENRE_WEIGHT * same_count)
        
        if np.isnan(cor):
            continue
        else:
            result.append((title, '{:.2f}'.format(cor), temp_genres))
            
    result.sort(key=lambda r: r[1], reverse=True)

    return result[:n]
    
    recommend_result = recommend('Best Seller', matrix, 10, similar_genre=True)

pd.DataFrame(recommend_result, columns = ['Title', 'Correlation', 'Genre'])


```
