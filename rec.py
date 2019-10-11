import math

#Class for Popularity based Recommender System model
class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.popularity_recommendations = None

    def smooth_user_preference(self, x):
        return math.log(1 + x, 2)

    #Create the popularity based recommender system model
    def create(self, train_data, user_id):
        self.train_data = train_data
        self.user_id = user_id

        event_type_strength = {
            'Followed': 1.0,
            'Like': 1.0,
            'Love': 2.0,
            'Commented': 4.0,
            'Replied': 4.0
        }
        train_data['eventStrength'] = train_data['action'].apply(lambda x: event_type_strength[x]).copy()
        users_interactions_count = train_data.groupby(['user_id', 'post_id']).size().groupby('user_id').size()
        users_with_enough_interactions = users_interactions_count[users_interactions_count >= 2].reset_index()[['user_id']]
        interactions_from_selected_users = train_data.merge(users_with_enough_interactions, how='right', left_on='user_id', right_on='user_id')
        interactions_full = interactions_from_selected_users.groupby(['user_id'])['eventStrength'].sum().apply(self.smooth_user_preference).reset_index()
        popular_users = interactions_full.sort_values('eventStrength', ascending=False)

        #Get the top 10 recommendations
        self.popularity_recommendations = popular_users.head(10)

    #Use the popularity based recommender system model to
    #make recommendations
    def recommend(self, user_id):
        user_recommendations = self.popularity_recommendations

        popular_users = user_recommendations[user_recommendations['user_id'] != user_id]

        # drop unnecessary columns
        popular_users.drop(['eventStrength'], axis=1, inplace=True)

        return popular_users
