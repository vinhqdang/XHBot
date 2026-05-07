import matplotlib.pyplot as plt
import numpy as np

methods = ['Standard GCN', 'XHBot (Ours)']
metrics = ['F1 Score', 'AUC-ROC', 'Accuracy', 'MCC']
gcn_scores = [0.6737, 0.9279, 0.9225, 0.6310]
xhbot_scores = [0.9474, 0.9879, 0.9875, 0.9420]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
rects1 = ax.bar(x - width/2, gcn_scores, width, label='Standard GCN', color='#d62728')
rects2 = ax.bar(x + width/2, xhbot_scores, width, label='XHBot (Ours)', color='#1f77b4')

ax.set_ylabel('Score')
ax.set_title('Performance Comparison on High-Noise Planted Heterophily Graph')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim([0, 1.1])
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('manuscript/figures/performance_comparison.png', bbox_inches='tight', dpi=300)
print('Figure saved to manuscript/figures/performance_comparison.png')
