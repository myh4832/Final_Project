import pandas as pd

result = pd.read_csv('./outputs/caption_4ds/result.csv')

score_sum = result[' B4'] + result[' M']

best_result = result.iloc[score_sum.idxmax()]

print(f'Best BLEU4 : {best_result[" B4"]}   Best METEOR : {best_result[" M"]}')