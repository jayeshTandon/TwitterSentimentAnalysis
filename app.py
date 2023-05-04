from flask import Flask, render_template, request, url_for, session
import prediction
import pandas as pd
import os
from config import API_SECRET

app = Flask(__name__)
app.secret_key = API_SECRET

@app.route("/")
def index():
    # Delete pos_wordcloud.png if it exists
    if os.path.exists("./static/pos_wordcloud.png"):
        os.remove("./static/pos_wordcloud.png")
    
    # Delete neg_wordcloud.png if it exists
    if os.path.exists("./static/neg_wordcloud.png"):
        os.remove("./static/neg_wordcloud.png")
    
    # Delete csv_file.csv if it exists
    if os.path.exists("csv_file.csv"):
        os.remove("csv_file.csv")

    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        keyword = request.form['keyword']
        tweetCounts = int(request.form['tweetCounts'])
        
        if len(keyword) > 0:
            prediction.setKeyword(keyword)
            result = prediction.prediction(count=tweetCounts)
            session['session_data'] = {'result': result}
        else:
            result = "Invalid Input"
            return render_template('index.html', error="Invalid Input")
        
    else:
        session_data = session.get('session_data')
        if session_data:
            result = session_data['result']
        else:
            result = "No Data"

    if os.path.exists('csv_file.csv'):
        if(os.path.exists('./static/pos_wordcloud.png')):
            pos_imgUrl = url_for('static', filename='pos_wordcloud.png')
        else:
            pos_imgUrl = url_for('static', filename='no_pos.jpg')

        if(os.path.exists('./static/neg_wordcloud.png')):
            neg_imgUrl = url_for('static', filename='neg_wordcloud.png')
        else:
            neg_imgUrl = url_for('static', filename='no_neg.jpg')

        return render_template('predict.html', pred=result, pos_wordcloud=pos_imgUrl, neg_wordcloud=neg_imgUrl)
    
    else:
        return render_template('index.html', error="No Tweets Found Related to the given Keyword")
        

@app.route('/tweets')
def tweets():
    if os.path.exists('csv_file.csv'):
        df = pd.read_csv('csv_file.csv')
        tweets = df.to_dict('records')
        return render_template('viewTweets.html', tweets=tweets)
    else:
        return render_template('viewTweets.html')


if __name__ == "__main__":
    app.run(debug=True)

