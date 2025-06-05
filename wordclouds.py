# python3 wordclouds.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.cm as cm
import jieba
import matplotlib
import os
import matplotlib.pyplot as plt

input_file = "LWC-messages/self_belief_messages_20250604.xlsx"
stopwords_file = "stopwords-zh.txt"
output_file = "wordclouds/"
num_components = 20
use_sklearn = False # sklearn is WAY faster

# Load the stopwords
with open(stopwords_file, 'r', encoding='utf-8') as f:
    stopwords_zh = set(f.read().splitlines())
    stopwords_zh = {word.strip() for word in stopwords_zh if word.strip()}  # Clean up any extra spaces
print(f"{len(stopwords_zh)} stopwords loaded successfully.")
    
df = pd.read_excel("/chronos_data/smangalik/weibo/LWC-messages/self_belief_messages_20250604.xlsx")

print("\nData loaded successfully.")
print(df)

# Tokenize the chinese text
def tokenize_chinese_text(text):
    tokens = jieba.cut(text)
    # Filter out punctuation and empty tokens
    tokens = [word for word in tokens if word.strip() and all(char.isalpha() or char.isspace() for char in word)]
    return tokens

# Apply the tokenizer to the 'text' column
df['tokens'] = df['text'].apply(tokenize_chinese_text)

# Create a document term matrix

vectorizer = CountVectorizer(tokenizer=lambda x: x, 
                             lowercase=False,
                             ngram_range=(1, 2), 
                             max_features=1000,
                             stop_words=stopwords_zh
                             )
dtm = vectorizer.fit_transform(df['tokens'])
# Convert the document term matrix to a DataFrame
dtm_df = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names_out())

print("\nDocument term matrix created successfully.")
print(dtm_df)

# print("\nTop 100 words by frequency:")
# top_words = dtm_df.sum().sort_values(ascending=False).head(100)
# print(top_words)

if use_sklearn: # Use sklearn.decomposition.FactorAnalysis
  fa = FactorAnalysis(rotation="varimax")
  fa.set_params(n_components=num_components)
  fa.fit(dtm_df)
  loadings = fa.components_.T
else: # Use factor_analyzer.FactorAnalyzer
  fa = FactorAnalyzer(rotation="varimax", n_factors=num_components, method='principal')
  fa.fit(dtm_df)
  loadings = fa.loadings_

# get the rotated factor pattern
factors = pd.DataFrame(loadings, index=dtm_df.columns, columns=[f"Factor_{i+1}" for i in range(num_components)])

# Create Factor Dictionary
factor_word_probability_dict = {}
for col in factors.columns:
  factor_word_probability_dict[col] = factors[col].to_dict()

# Print Factors
factor_df_list = []
print("\nFactors:\n")
for _factor, _word_probability_dict in factor_word_probability_dict.items():
    print(_factor)
    _word_probability_dict_sorted = sorted(_word_probability_dict.items(), key=lambda x: x[1], reverse=True)
    for i, (_word, _probability) in enumerate(_word_probability_dict_sorted):
      if i < 10:
        print(" ->",round(_probability, 4), '\t', _word)
      factor_df_list.append({
          'category':int(_factor.split("_")[1]),
          'term':_word,
          'weight':_probability
      })
    print()

# Display the factorization
factor_df = pd.DataFrame(factor_df_list)
print("\nFactors retrieved")
print(factor_df)

print("\nCreating word clouds for each factor...\n")
# Delete contents of wordclouds/
if not os.path.exists('wordclouds'):
    os.makedirs('wordclouds')
for filename in os.listdir('wordclouds'):
    file_path = os.path.join('wordclouds', filename)
    try:
        os.unlink(file_path)
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')

# Color Function
def my_tf_color_func(word_freqs):
  norm = matplotlib.colors.Normalize(vmin=min(word_freqs.values()), vmax=max(word_freqs.values()))
  m = cm.ScalarMappable(norm=norm, cmap="Blues")
  def my_tf_color_func_inner(word, font_size, position, orientation, random_state=None, **kwargs):
    rgb = m.to_rgba(word_freqs[word])[:3]
    color = matplotlib.colors.rgb2hex(rgb)
    # print(word, font_size, position, word_freqs[word], color) #Debugging
    return color
  return my_tf_color_func_inner

print("\Making Clouds...")
for _topic, _word_probability_dict in factor_word_probability_dict.items():
  # Generate a word cloud image
  wordcloud = WordCloud(max_words=50, color_func=my_tf_color_func(_word_probability_dict),background_color='white').fit_words(_word_probability_dict)
  plt.imshow(wordcloud, interpolation='bilinear');
  plt.title(_topic)
  plt.axis("off");
  plt.savefig('wordclouds/'+str(_topic) + '.png')