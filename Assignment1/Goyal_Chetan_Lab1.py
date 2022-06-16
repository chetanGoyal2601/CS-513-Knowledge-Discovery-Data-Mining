import numpy as np
import pandas as pd
df = pd.read_csv(r'breast-cancer-wisconsin.csv')

#Answer 1.1
df.describe().to_csv("Assignment1.csv")

