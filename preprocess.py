data = pd.read_csv('D:\VScode\naanmudhalvangenai\emotion_sentimen_dataset.csv')
missing_values = data.isnull().sum()
print(missing_values)

# Remove rows with missing values in the "Emotion" column
data = data.dropna(subset=['Emotion'])
data.reset_index(drop=True, inplace=True)
print(data.head())

data = data.drop_duplicates()

# Text cleaning
data['clean_text'] = data['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Remove special characters
data['clean_text'] = data['clean_text'].apply(lambda x: re.sub(r'\d+', '', x))  # Remove numbers
data['clean_text'] = data['clean_text'].apply(lambda x: x.lower())  # Convert text to lowercase