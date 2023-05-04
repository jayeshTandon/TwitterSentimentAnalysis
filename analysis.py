# from io import BytesIO
import pickle
# import pandas as pd
from preprocessing_methods import preProcess
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def analyseSentiment(data):
    with open('model_svm.pkl', 'rb') as f:
        classifer,cv = pickle.load(f)
    
    data = preProcess(data)
    data_transformed = cv.transform(data['text']).toarray()    
    pred = classifer.predict(data_transformed)

    makeWordCloud(pred, data)
    return pred

def makeWordCloud(pred, data): 
    pos_word = []
    neg_word = []
    for i in range(len(pred)):
        if pred[i] == 1:
            pos_word.extend([word for word in data.iloc[i]['text'].split()])
        else: 
            neg_word.extend([word for word in data.iloc[i]['text'].split()])
    
    pos_words = " ".join(pos_word)
    neg_words = " ".join(neg_word)
    

    # Positive Cloud
    try:
        if len(pos_word) == 0:
            raise Exception("No Positive Tweets Found.")
    
        
        pos_wordcloud = WordCloud(width=800, height=512, random_state=42, max_font_size=100, stopwords=["andamp"], collocations=False).generate(pos_words)
        plt.figure(figsize=(15,8))
        plt.imshow(pos_wordcloud, interpolation='bilinear')
        plt.axis('off')

        #Saving the Positive Cloud
        pos_wordcloud.to_file('./static/pos_wordcloud.png')
    except Exception as e:
        print("Error ::", e)
    
    #Negative Cloud
    try:
        if len(neg_word) == 0:
            raise Exception("No Negative Tweets Found.")
        
        neg_wordcloud = WordCloud(width=800, height=512, random_state=42, max_font_size=100, stopwords=["andamp"], collocations=False).generate(neg_words)
        plt.figure(figsize=(15,8))
        plt.imshow(neg_wordcloud, interpolation='bilinear')
        plt.axis('off')

        #Saving the Negative Cloud
        neg_wordcloud.to_file('./static/neg_wordcloud.png')
    except Exception as e:
        print("Error ::", e)
