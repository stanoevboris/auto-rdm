from scipy.stats import friedmanchisquare

# BINARY
# F1
woe_f1 = [0.338, 0.821, 0.535, 0.622, 0.876, 0.756]
word_MiniLM_f1 = [0.364, 0.821, 0.815, 0.667, 0.898, 0.688]
word_tf_idf_f1 = [0.325, 0.821, 0.805, 0.833, 0.898, 0.727]

# ROC AUC
woe_auc = [0.718, 0.738, 0.499, 0.662, 0.899, 0.550]
word_MiniLM_auc = [0.762, 0.709, 0.905, 0.633, 0.935, 0.545]
word_tf_idf_auc = [0.705, 0.746, 0.866, 0.833, 0.920, 0.670]

stat, p = friedmanchisquare(woe_f1, word_MiniLM_f1, word_tf_idf_f1)
print(f'Friedman test statistic for F1-binary: {stat}, p-value: {p}')

stat, p = friedmanchisquare(woe_auc, word_MiniLM_auc, word_tf_idf_auc)
print(f'Friedman test statistic for ROCAUC-binary: {stat}, p-value: {p}')

woe_f1_mc = [0.934, 0.336, 0.961, 0.927, 0.999, 0.693, 0.52, 0.952]
word_MiniLM_f1_mc = [0.318, 0.426, 0.41, 0.639, 0.393, 0.96, 0.55, 0.635]
word_tf_idf_f1_mc = [0.139, 0.403, 0.926, 0.691, 0.701, 0.977, 0.528, 0.886]

stat, p = friedmanchisquare(woe_f1_mc, word_MiniLM_f1_mc, word_tf_idf_f1_mc)
print(f'Friedman test statistic for F1-multiclass: {stat}, p-value: {p}')
