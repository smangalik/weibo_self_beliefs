# python3 data_self_beliefs.py

import pandas as pd
import warnings
from deep_translator import GoogleTranslator
from tqdm import tqdm
from datetime import datetime
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'


input_file = 'LWC-messages/cleaned_messages.csv'
output_file = f'LWC-messages/self_belief_messages_{datetime.now().strftime("%Y%m%d")}.xlsx'
sample_size = 25

print(f"\nReading cleaned messages from {input_file}")
df = pd.read_csv(input_file, encoding='utf-8', nrows=None)

print(f"\nColumns: {df.columns.tolist()}")
print(f"Initial DataFrame shape: {df.shape}")
print(df[['message_id','text']])

# Message must contain 我 (wǒ) or 俺 (ǎn) to be considered as a self-belief
df = df[df['text'].str.contains(r'我|俺', na=False, regex=True)]
print(f"\nFiltered DataFrame shape after requiring 我 (wǒ) or 俺 (ǎn): {df.shape}")
print(df[['message_id','text']])

# Create regex patterns to match the above phrases
self_belief_patterns = [
    # High
    r'(?:我|俺)觉得.*我是.*的人.*',     # Wǒ juéde… wǒ shì… de ren (I feel… I am… kind of person)
    r'(?:我|俺)觉得.*我们是.*的人.*',   # Wǒ juéde… wǒ men shì… de ren (I feel… we are… kind of people)
    # Medium
    r'(?:我|俺)是.*的人.*',             # Wǒ shì… de rén (I am a person who is…)
    r'(?:我|俺)们是.*的人.*',           # Wǒ men shì… de rén (I am a people who are…)
    r'(?:我|俺)觉得.*我是.*',           # Wǒ juéde… wǒ shì… (I feel… I am…)
    r'(?:我|俺)觉得.*我们是.*',         # Wǒ juéde… wǒ men shì… (I feel… we are…)
    r'(?:我|俺)觉得我很.*',             # Wǒ juéde wǒ hěn… (I think I am very…)
    r'(?:我|俺)觉得我们很.*',           # Wǒ juéde wǒ men hěn… (I think we are very…)
    # Low
    # r'(?:我|俺)是.*',                 # Wǒ shì… (I am…)
    # r'(?:我|俺)相信我.*'              # Wǒ xiāngxìn wǒ… (I believe I am…)
    # Bad
    #r'(?:我|俺)有点.*',                # Wǒ yǒudiǎn… (I have a bit of…)
]

def translate_to_english(texts):
    translator = GoogleTranslator(source='zh-CN', target='en')
    translations = []
    if isinstance(texts, str):
        return translator.translate(texts)
    else:
        for text in tqdm(texts, desc="Translating to English", unit="sentence"):
            try:
                translation = translator.translate(text)
                translations.append(translation)
            except Exception as e:
                print(f"Error translating {text}: {e}")
                translations.append("")
    return translations

print()
df_self_beliefs = []
df_self_beliefs_sample = []
for pattern in self_belief_patterns:
    df_filtered = df[df['text'].str.contains(pattern, na=False, regex=True)]
    df_filtered['pattern'] = pattern
    print(f"Found {len(df_filtered)} messages matching pattern: {pattern}")
    
    df_self_beliefs.append(df_filtered)
    
    if len(df_filtered) >= sample_size:
        df_filtered_sample = df_filtered.sample(sample_size, random_state=42)  
    else:
        df_filtered_sample = df_filtered
    df_filtered_sample['text_en'] = translate_to_english(df_filtered_sample['text'])
    
    df_self_beliefs_sample.append(df_filtered_sample)
    
df_self_beliefs = pd.concat(df_self_beliefs, ignore_index=True).drop_duplicates(subset=['text'])
df_self_beliefs_sample = pd.concat(df_self_beliefs_sample, ignore_index=True).drop_duplicates(subset=['text'])

print(f"\nFiltered messages containing self belief pattern. Final shape: {df_self_beliefs.shape}")
print(df_self_beliefs[['message_id','pattern','text']])

print(f"\nSaving self belief messages to {output_file}...")
df_self_beliefs.to_excel(output_file, index=False, encoding='utf-8')

print(f"\nSaving self belief sample messages to {output_file.replace('.xlsx', '_sample.xlsx')}...")
df_self_beliefs_sample[['message_id','pattern','text','text_en']].to_excel(output_file.replace('.xlsx', '_sample.xlsx'), index=False, encoding='utf-8')