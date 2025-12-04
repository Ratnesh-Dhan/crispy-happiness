import pandas as pd

input_file = "/mnt/d/DATASETS/main_Noamundi_data.xlsx"
output_file = "/mnt/d/DATASETS/distinct_Noamundi_data.xlsx"
column_name = "Sample No"   # change this to the column you want

# read the data
df = pd.read_excel(input_file)

# drop duplicates based on the given column
df_unique = df.drop_duplicates(subset=[column_name], keep="first")

# write to new file
df_unique.to_excel(output_file, index=False)

print("Done. Check output.xlsx")
