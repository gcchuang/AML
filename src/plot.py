# %%
import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Load the CSV file into a pandas DataFrame
df = pd.read_csv('/home/weber50432/AML_image_processing/MIL_slide_level/output/DNMT3A/predictions.csv', sep=",", encoding="utf-8")
# Get the true labels and predicted labels
y_true = df['target']
y_pred = df['prediction']
# Get the gene mutation probabilities
y_score = df['probability']
cm = metrics.confusion_matrix(y_true, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])

cm_display.plot()
plt.show()








