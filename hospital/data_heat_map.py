import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

df = pd.read_csv(f'hospital/org_train_data.csv')

fig, ax = plt.subplots(1, 1, figsize=(7, 3))
sb.heatmap(df.isnull(), cbar=False)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Variables')
ax.set_ylabel('Observations')
fig.tight_layout()
plt.show()
