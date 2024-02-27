import pandas as pd
import re 
import string
from spellchecker import SpellChecker 

spell = SpellChecker()
# type text split id 
dataset = pd.read_csv("./datasets/kaggle.csv")
dataset.set_index('type', inplace=True)

# set the personality to take from 
personality = 'INFJ'
# 8675
print(len(dataset))

def fix_typos(text):
    # Split text into words
    words = text.split()
    
    # Iterate over each word
    corrected_words = []
    for word in words:
        # Check if word is misspelled
        corrected_word = spell.correction(word)
        if corrected_word is not None: 
            corrected_words.append(corrected_word)
    
    # Join corrected words back into a single string
    corrected_text = ' '.join(corrected_words)
    return corrected_text

def cleaning_data(text):
    '''Remove web url'''
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', str(text), flags=re.MULTILINE)
    '''Make text lowercase'''
    text = text.lower()
    '''remove text in square brackets'''
    text = re.sub('\[.*?\]', '', text)
    '''remove punctuations'''
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    '''remove digits'''
    text = re.sub('\w*\d\w*', '', text)
    # '''remove stop words'''
    # STOPWORDS = set(stopwords.words('english'))
    # text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    
    '''Get rid of some additional punctuations '''
    text = re.sub('\[''""...]', '', text)
    '''Get rid of non-sensical'''
    text = re.sub('\n', '', text)
    '''Remove single characters from the start'''
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    '''Removing prefixed 'b'''
    text = re.sub(r'^b\s+', '', text)
    '''Get rid of all single characters'''
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', str(text))
    '''Remove all the special characters'''
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
    
    return text

# remove extra white spaces
def remove_extra_spaces(text):
    text = re.sub(r'\s+', ' ', text)

    return text

def append_word(text):
    return text + ' ' + '<EOS>'

dataset = dataset.loc[personality]
sentence_series = dataset['text'].str.split('\|\|\|')

expanded_dataset = sentence_series.explode() 
expanded_dataset = expanded_dataset.to_frame(name='text')
# 422844
print(len(expanded_dataset))

expanded_dataset['text'] = expanded_dataset['text'].apply(cleaning_data)
# expanded_dataset['text'] = expanded_dataset['text'].apply(fix_typos)

# joins all the text, splits it into individual words then counts the frequency of each word 
# [-500:] selects the 500 least common words 
freq = pd.Series(' '.join(expanded_dataset['text']).split()).value_counts()[-500:]
# let's remove these words as their presence will be of any use
freq = list(freq.index)

print(freq)
'''Remove rare words'''
expanded_dataset['text'] = expanded_dataset['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

expanded_dataset = expanded_dataset[expanded_dataset['text'].str.strip() != '']

min_length = 3
max_length = 15
expanded_dataset = expanded_dataset[expanded_dataset['text'].str.split().apply(lambda x: min_length <= len(x) <= max_length)]
print(len(expanded_dataset))
expanded_dataset['text'] = expanded_dataset['text'].apply(remove_extra_spaces)
expanded_dataset['text'] = expanded_dataset['text'].apply(append_word)
print(expanded_dataset[:100])

# write to a file base on the personality type
# INFJ = expanded_dataset.loc['INFJ']
file_path = f'./MBTI/{personality}.txt'
expanded_dataset['text'].to_csv(file_path, index=False, header=False, sep='\n')