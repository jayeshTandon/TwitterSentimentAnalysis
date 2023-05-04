# Removing special characters and symbols: Some tweets may contain special characters and symbols that are not relevant to the sentiment analysis and can be removed. This can include hashtags, mentions, and URLs.
# Removing stop words: Stop words are common words that do not provide much meaning in the context of sentiment analysis. These words can be removed to reduce the dimensionality of the data and increase the accuracy of the analysis.
# Lowercasing text: To avoid any confusion between uppercase and lowercase words, it's a good practice to convert all text to lowercase.
# Removing punctuation: Punctuation marks may not be relevant for sentiment analysis and can be removed.
# Tokenizing: Tokenizing is the process of breaking down the text into individual words or phrases. This is an important step for sentiment analysis, as it allows for the identification of key words and phrases that may be indicative of sentiment.
# Stemming or Lemmatizing: Stemming is the process of reducing inflected words to their word stem, base or root form, so that words with similar meaning will be reduced to the same form, while Lemmatizing is the process of reducing a word to its base form. This step is important to reduce the dimensionality of the data and increase the accuracy of the analysis.
# Removing Emoji and emoticons: Emoji and emoticons can carry a lot of sentiment information, but they are not always easy to process, it's best to remove them if you want to focus on the text sentiment only.
# Removing elongated words: As discussed before, elongated words are words that have repeated letters, usually used to indicate emphasis or excitement. These words may not be useful for the analysis and can be removed.
# Removing HTML tags and XML tags: Some tweets may contain HTML or XML tags, which are not relevant for sentiment analysis and can be removed.
# Handling hashtags: Hashtags can carry a lot of information, but they should be handled carefully. Some hashtags are used to indicate the topic of the tweet, but others may be used to indicate sentiment. Identifying the sentiment of hashtags can be useful for sentiment analysis.

import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from spellchecker import SpellChecker

nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def removeUnicode(text):
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    return text 

def lowerCase(text):
    text = text.lower()
    return text

def removeUrl(text):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',r'',text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text

def removeAtMention(text):
    text = re.sub('@[^\s]+',r'',text)
    return text

def removeHashtags(text):
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text

def removeNumbers(text):
    text = ''.join([i for i in text if not i.isdigit()])         
    return text

def removeExtraExclamationMarks(text):  
    text = re.sub(r"(\!)\1+", '!', text)
    return text

def removeExtraQuestionMarks(text):
    text = re.sub(r"(\?)\1+", '?', text)
    return text

def removeExtraPeriods(text):
    text = re.sub(r"(\.)\1+", '.', text)
    return text

def replaceContraction(text):
    contraction_patterns = [ (r'won\'t', 'will not'), (r'y\'all', 'you all'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                         (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text

def handleElongatedWords(sentence):
    elongated_words = re.findall(r'\b(\w*(\w)\2\w*)\b', sentence)
    for word in elongated_words:
        initial_word = word[0]
        if initial_word.lower() not in words.words():
            first = initial_word.index(word[1])
            for i in range(first, (len(initial_word)-1)):
                if initial_word[i] == initial_word[i+1]:
                    check_word = initial_word[:first] + initial_word[i+1:]
                    if check_word.lower() in words.words():
                        sentence = sentence.replace(initial_word, check_word)
                        break   
    return sentence

def removeStopWords(sentence):
    stop_words = set(stopwords.words("english"))
    stop_words.difference_update(["no", "not", "neither", "very"])
    words = word_tokenize(sentence)
    filtered_sentence = []
    for word in words:
        if word.lower() not in stop_words:
            filtered_sentence.append(word)
    return " ".join(filtered_sentence)

def lemmatizer(sentence):
    lemmatizer = WordNetLemmatizer()
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tags = {'N': 'n', 'V': 'v', 'R': 'r'}
    words = [lemmatizer.lemmatize(word, wordnet_tags.get(tag[0], 'n')) for word, tag in nltk_tagged]
    return " ".join(words)

def preProcess(df):
    df = df.copy()
    for index in df.index:
        text = df['text'][index] 

        text = removeUnicode(text) #as well as emoji(s)
        text = text.lower()
        text = removeUrl(text)
        text = removeAtMention(text)
        text = removeHashtags(text)
        text = removeNumbers(text)
        text = removeExtraExclamationMarks(text)
        text = removeExtraQuestionMarks(text)
        text = removeExtraPeriods(text)
        text = replaceContraction(text)
        text = handleElongatedWords(text)
        text = removeStopWords(text)
        text = lemmatizer(text)
        
        df.loc[index,'text'] = text
    return df