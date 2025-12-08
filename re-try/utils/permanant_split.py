import pandas as pd
import os

base_path = "/mnt/d/DATASETS/noamundi_split_dataset"
df = pd.read_excel(os.path.join(base_path, "distinct_Noamundi_data.xlsx"))

# Pick random 20 rows
test_df = df.sample(n=20, random_state=42)

# Remove those 20 rows from the main dataset
train_df = df.drop(test_df.index)

# Save both
test_df.to_excel(os.path.join(base_path, "final_test_dataset.xlsx"), index=False)
train_df.to_excel(os.path.join(base_path, "training_dataset.xlsx"), index=False)
