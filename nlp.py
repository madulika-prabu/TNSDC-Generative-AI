from sklearn.model_selection import train_test_split
import nltk
nltk.download('punkt')

data['tokens'] = data['text'].apply(word_tokenize) #Tokenization

data['tokens'] = data['tokens'].apply(lambda x: [word.lower() for word in x if word.isalnum()]) #Text Normalization

#Stopword Removal
stop_words = set(stopwords.words('english'))
data['tokens'] = data['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

# Stemming
stemmer = PorterStemmer()
data['tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(word) for word in x])

data['clean_text'] = data['tokens'].apply(lambda x: ' '.join(x)) #Convert tokens back to text

X_train, X_test, y_train, y_test = train_test_split(data['clean_text'], data['Emotion'], test_size=0.2, random_state=42)

missing_indices = y_train[y_train.isnull()].index
X_train = X_train.drop(index=missing_indices)
y_train = y_train.drop(index=missing_indices)
print("NaN values in y_train:", y_train.isnull().sum())

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)