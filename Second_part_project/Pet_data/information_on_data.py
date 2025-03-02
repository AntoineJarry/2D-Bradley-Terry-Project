
import pandas as pd
# Charger le fichier Excel
cat_df = pd.read_excel("AMA08001.xlsx")
dog_df = pd.read_excel("VYE15004.xlsx")


# Function to print unique values for each column in a DataFrame and check NA counts
def print_unique_values(df, dataset_name):
    print(f"\nUnique values in {dataset_name}:")
    for col in df.columns:
        unique_vals = df[col].dropna().unique()  # unique non-NA values
        na_count = df[col].isna().sum()           # count NA values
        print(f"\nColumn '{col}' ({len(unique_vals)} unique non-NA value(s), NA count: {na_count}):")
        print(unique_vals)

# Print unique values for all columns in the Dog dataset
print_unique_values(dog_df, "Dog Dataset")

# Print unique values for all columns in the Cat dataset
print_unique_values(cat_df, "Cat Dataset")