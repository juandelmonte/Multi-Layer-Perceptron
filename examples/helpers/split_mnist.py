import pandas as pd
import os

# Path to the original mnist_train.csv
currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
filename = os.path.join(parentdir, 'data', 'mnist_train.csv')

# Read the data
df = pd.read_csv(filename, header=0)

# Shuffle the dataframe
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train (90%) and validation (10%)
total_rows = len(df_shuffled)
validation_size = int(total_rows * 0.1)
train_size = total_rows - validation_size

df_train = df_shuffled[:train_size]
df_validation = df_shuffled[train_size:]

# Write to new files in examples/data/
data_dir = os.path.join(currentdir, 'data')
os.makedirs(data_dir, exist_ok=True)

train_filename = os.path.join(data_dir, 'mnist_train_split.csv')
validation_filename = os.path.join(data_dir, 'mnist_validation.csv')

df_train.to_csv(train_filename, index=False)
df_validation.to_csv(validation_filename, index=False)

print(f"Original dataset size: {total_rows}")
print(f"Train split size: {len(df_train)}")
print(f"Validation split size: {len(df_validation)}")
print(f"Files saved: {train_filename}, {validation_filename}")