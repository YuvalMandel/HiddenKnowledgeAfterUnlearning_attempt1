import pandas as pd, numpy as np

df = pd.read_parquet('C:/technion/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1/plots/base_k_scores.parquet')
f = df[(df['domain']=='bio') & (df['probe_type']=='own') & (df['clf']=='LR') & (df['layer_config']=='full')]
n = len(f)
ki = f['k_internal'].values
ke = f['k_external'].values
ke_r = (ke * 3).round() / 3

print(f'N={n}')
print(f'K_ext mean:            {ke.mean()*100:.1f}%')
print(f'K_ext accuracy (>0.5): {(ke>0.5).mean()*100:.1f}%')
print(f'K_int accuracy (>0.5): {(ki>0.5).mean()*100:.1f}%')
print(f'Test AUC (probe):      {f["test_auc"].iloc[0]*100:.1f}%')
print(f'Val AUC  (probe):      {f["val_auc"].iloc[0]*100:.1f}%')
for v, lbl in [(0, '0'), (1/3, '1/3'), (2/3, '2/3'), (1.0, '1')]:
    c = int((np.abs(ke_r - v) < 0.01).sum())
    print(f'  K_ext={lbl}: {c} ({c/n*100:.1f}%)')
