import pandas as pd
df1 = pd.read_csv('DataPart01.csv')
df2 = pd.read_csv('DataPart02.csv')

frames = [df1, df2]
all_csv = pd.concat(frames)
all_csv.to_csv('DatasetAll.csv')