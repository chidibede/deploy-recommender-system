''' Deploy user recomender model using flask framework'''

# Import Relevant Modules
try:
    import pickle
    import joblib
    from sklearn.metrics.pairwise import linear_kernel
    from flask import Flask, jsonify
    from flask import request, render_template
    import pandas as pd
    from pandas.io.json import json_normalize

except ImportError as i_error:
    print(i_error)

# Load the datasets needed for deployment
USERS = pd.read_csv('used_data/users.csv')
USERS['name'] = USERS.name.str.lower()
USERS_SIM = pd.read_csv('used_data/users_sim.csv')
USERS_SIM['name'] = USERS_SIM.name.str.lower()
POSTS = pd.read_csv('used_data/posts_deploy.csv', index_col=0)
INDICES = pd.Series(USERS_SIM['user_id'].index)

# load the model from disk
MODEL = joblib.load('popular.sav')
SIMILAR_MODEL = pickle.load(open('finalized_model.sav', 'rb'))
ARTICLE_MODEL = pickle.load(open('final_model.sav', 'rb'))

# computing TF-IDF matrix required for calculating cosine similaritIES
USERS_TRANSF = SIMILAR_MODEL.fit_transform(USERS_SIM['short_bio'])
COSINE_SIMILARITY = linear_kernel(USERS_TRANSF, USERS_TRANSF)
POSTS_TRANSF = ARTICLE_MODEL.fit_transform(POSTS['title'])
COS_SIMILARITY = linear_kernel(POSTS_TRANSF, POSTS_TRANSF)

def recommend(index, cosine_sim=COSINE_SIMILARITY):
    '''Declaring a function that would use our model to fetch users
    similar to a given user based on user_bio'''
    try:
        i_d = INDICES[index]
        # Get the pairwsie similarity scores of all names
        # sorting them and getting top 10
        similarity_scores = list(enumerate(cosine_sim[i_d]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:11]

        # Get the names index
        users_index = [i[0] for i in similarity_scores]

        # Return the top 10 most similar names
        return USERS_SIM['name'].iloc[users_index]
    except KeyError:
        return 'Invalid User ID, Enter a valid User Id'
    except IndexError:
        return 'This user has no bio'

def post_recommend(index, cosine_sim=COS_SIMILARITY):
    '''Function to recommend articles to users'''
    try:
        i_d = INDICES[index]
        # Get the pairwsie similarity scores of all names
        # sorting them and getting top 10
        similarity_scores = list(enumerate(cosine_sim[i_d]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:11]

        # Get the names index
        posts_index = [i[0] for i in similarity_scores]

        # Return the top 10 most similar names
        return POSTS['content'].iloc[posts_index]
    except KeyError:
        return 'This user has no bio description/no article recommendation'
    except IndexError:
        print("")

# Initialize the app
app = Flask(__name__)

'''HTML GUI User Recommendation'''

# Render the home page
@app.route('/')
def home():
    '''Display of web app homepage'''
    return render_template('index.html')

# render the new_user_recommend page
@app.route('/new_user_recommend')
def new_user_recommend():
    '''Display of new user recommendation page'''
    return render_template('new_user_recommend_form.html')

# render the similar_user_recommend page
@app.route('/similar_user_recommend')
def similar_user_recommend():
    '''Display of similar user recommendation page'''
    return render_template('similar_user_recommend_form.html')

@app.route('/article_user_recommend')
def article_user_recommend():
    '''Display of similar user recommendation page'''
    return render_template('article_recommend_form.html')

# render the new user recommend page
@app.route('/recommend', methods=['POST'])
def new_user_recommender():
    '''Function that accepts the features and predicts
    a response(user recommendation) and displays it
    for Web App Testing'''
    # get the values from the form
    try:
        name_of_user = [x for x in request.form.values()]
        for name in name_of_user:
            name_of_user = name.lower()

        # recommend
        popular_users = MODEL.recommend((USERS[USERS.name == name_of_user]['name']).index[0])

        recommended_users = []
        for i_d in popular_users['user_id']:
            recommended_users.append(USERS.iloc[i_d, 0])
        return render_template('recommend.html', prediction_text=recommended_users)
    except:
        return render_template('recommend.html', prediction_text=["User does not exist"])

# render the recommended results
@app.route('/similar_recommend', methods=['POST'])
def similar_user_recommender():
    '''Function that accepts the features and predicts
    a response(user recommendation) and displays it
    for Web App Testing'''
    try:
        # get the values from the form
        name_of_user = [x for x in request.form.values()]
        for name in name_of_user:
            name_of_user = name.lower()
        # recommend
        similar_users = recommend(int(USERS_SIM[USERS_SIM.name == name_of_user]['user_id']))
        return render_template('similar_recommend.html', prediction_text=similar_users)
    except:
        text = ["User does not exist/Has no bio"]
        return render_template('similar_recommend.html', prediction_text=text)

# render the recommended results
@app.route('/post_recommend', methods=['POST'])
def article_user_recommender():
    '''Function that accepts the features and predicts
    a response(user recommendation) and displays it
    for Web App Testing'''
    try:
        # get the values from the form
        name_of_user = [x for x in request.form.values()]
        for name in name_of_user:
            name_of_user = name.lower()
        # recommend
        recommended_posts = post_recommend((USERS[USERS.name == name_of_user]['name']).index[0])
        return render_template('article_recommend.html', prediction_text=recommended_posts)
    except KeyError:
        return 'This user has no bio description/no article recommendation'
    except IndexError:
        return render_template('article_recommend.html', prediction_text=["User does not exist"])

@app.route("/similar_user_recommend_api", methods=['POST'])
def similar_user_recommend_api():
    '''Function that handles direct api calls
    from another client to recommend similar users'''
    try:
        json_data = request.get_json(force=True)
        json_df = json_normalize(json_data)
        for name in json_df.name:
            name_of_user = name.lower()
        recommended_users = recommend(int(USERS_SIM[USERS_SIM.name == name_of_user]['user_id']))
        recommended_users = {
            "recommended_users": [x for x in recommended_users]
        }
        return jsonify(recommended_users)
    except:
        print("http error")

@app.route("/new_user_recommend_api", methods=['POST'])
def new_user_recommend_api():
    '''Function that handles direct api calls
    from another client to recommend Most Popular users'''
    try:
        json_data = request.get_json(force=True)
        json_df = json_normalize(json_data)
        for name in json_df.name:
            name_of_user = name.lower()
        popular_users = MODEL.recommend((USERS[USERS.name == name_of_user]['name']).index[0])

        recommended_users = []
        for i_d in popular_users['user_id']:
            recommended_users.append(USERS.iloc[i_d, 0])
        recommended_users = {
            "recommended_users": [x for x in recommended_users]
        }
        return jsonify(recommended_users)
    except:
        print("Server error")

@app.route("/article_recommend_api", methods=['POST'])
def article_user_recommend_api():
    '''Function that handles direct api calls
    from another client to recommend articles'''
    json_data = request.get_json(force=True)
    json_df = json_normalize(json_data)
    for name in json_df.name:
        name_of_user = name.lower()
    recommended_posts = post_recommend(int(USERS_SIM[USERS_SIM.name == name_of_user]['user_id']))
    recommended_posts = {
        "recommended_posts": [x for x in recommended_posts]
    }
    return jsonify(recommended_posts)

# run the app
if __name__ == "__main__":
    app.run(debug=True)
