import pandas as pd
import os

csv_file_path = 'DatasetIPC.csv' 
df = pd.read_csv(csv_file_path)

output_dir = './MulitipleCountry/'
os.makedirs(output_dir, exist_ok=True)

df_empty_priority = pd.DataFrame(columns=df.columns)

for index, row in df.iterrows():
    priority_countries = row['priorityCountryRegion']
    ipc_value = row['IPC']  # 假设每行数据都有IPC列

    if pd.isnull(priority_countries) or priority_countries == '-':
        df_empty_priority = df_empty_priority.append({'IPC': ipc_value}, ignore_index=True)
    else:
        priority_countries_list = list(set(priority_countries.split(' | ')))

        for country in priority_countries_list:
            filename = f"{output_dir}IPC{country}.csv"
            # 使用DataFrame来存储数据，以便只包含IPC列
            df_temp = pd.DataFrame([ipc_value], columns=['IPC'])
            if os.path.exists(filename):
                df_temp.to_csv(filename, mode='a', header=False, index=False)
            else:
                df_temp.to_csv(filename, mode='w', header=True, index=False)

if not df_empty_priority.empty:
    df_empty_priority.to_csv(f"{output_dir}IPC-.csv", index=False)