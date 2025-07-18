import pandas as pd
import re
input_path = ".\合并结果.xlsx"
output_path = ".\processed_data.csv"

df = pd.read_excel(input_path)

def clean(text):
    if not isinstance(text, str):
        return text
    text = text.replace("nan", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["Product Details"] = df["Product Details"].apply(clean)

df.to_csv(output_path, index=False)








