# %%
import re
import os
import numpy as np
import pandas as pd

for root, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.parquet'):
            fpath = os.path.join(root, file)
            df = pd.read_parquet(fpath)
            del df['index']
            df.to_parquet(fpath, index=False)
