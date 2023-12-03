import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'User1': [4, 5, 0, 1, 0, 6],
    'User2': [6, 5, 0, 4, 2, 1],
    'User3': [0, 3, 6, 5, 0, 2],
    'User4': [2, 0, 1, 4, 6, 5],
}

df = pd.DataFrame(data, index=['Movie1', 'Movie2', 'Movie3', 'Book1', 'Book2','Book3'])


def get_similar_users(user, df):
    user_ratings = df.loc[:, user].values.reshape(1, -1)
    other_users = df.drop(columns=user).values.T  

    similarities = cosine_similarity(user_ratings, other_users)[0]

    similar_users = pd.Series(similarities, index=df.drop(columns=user).columns, name='Similarity')
    similar_users = similar_users.sort_values(ascending=False)

    return similar_users


def recommend_items(user, df, num_recommendations=3):
    similar_users = get_similar_users(user, df)

    recommendations = []
    for other_user in similar_users.index:
        other_user_ratings = df.loc[:, other_user]

        unrated_items = other_user_ratings[other_user_ratings == 0].index

        recommendations.extend(unrated_items[:num_recommendations])

    return recommendations[:num_recommendations]

all_recommendations = {}
for user in df.columns:
    recommendations = recommend_items(user, df)
    all_recommendations[user] = recommendations

for user, recommendations in all_recommendations.items():
    print(f"Recommended items for {user}: {recommendations}")
