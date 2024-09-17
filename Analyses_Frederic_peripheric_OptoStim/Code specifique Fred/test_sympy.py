from sympy import solve, symbols, Eq, exp, N, GreaterThan
import numpy as np
import matplotlib.pyplot as plt
import random

cell_Text = [] # Le contenu de la table
col_Labels = ["Stimulation Intensité", "Force maximale (%)"]
forceMaxPourc = range(0,110,10)
        
for val in forceMaxPourc:
    valeur = val*random.randint(1,100)
    cell_Text.append([str(valeur), str(val)])
print(cell_Text)
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis("off")
ax.axis("tight")

ax.table(cellText = cell_Text, colLabels=col_Labels, loc="top")
fig.tight_layout()
plt.show()

print()