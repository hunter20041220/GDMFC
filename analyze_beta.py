import pandas as pd

# Read Excel file
df = pd.read_excel('beta1-1000.xlsx')
df_clean = df[['beta', 'ACC', 'NMI', 'Purity']].dropna()

# Top 20 by ACC
top20_acc = df_clean.nlargest(20, 'ACC')
print('=== Top 20 Beta by ACC ===')
print(top20_acc.to_string(index=False))

# Best results
max_acc_idx = df_clean['ACC'].idxmax()
max_nmi_idx = df_clean['NMI'].idxmax()
max_pur_idx = df_clean['Purity'].idxmax()

print('\n=== Best Results ===')
print(f"Max ACC: {df_clean.loc[max_acc_idx, 'ACC']:.2f}% at beta={df_clean.loc[max_acc_idx, 'beta']:.0f}")
print(f"Max NMI: {df_clean.loc[max_nmi_idx, 'NMI']:.2f}% at beta={df_clean.loc[max_nmi_idx, 'beta']:.0f}")
print(f"Max Purity: {df_clean.loc[max_pur_idx, 'Purity']:.2f}% at beta={df_clean.loc[max_pur_idx, 'beta']:.0f}")

# Find good beta ranges (ACC >= 84)
good_betas = df_clean[df_clean['ACC'] >= 84.0]
print(f'\n=== Beta values with ACC >= 84% ({len(good_betas)} results) ===')
print(good_betas.to_string(index=False))

# Save to CSV for easy access
good_betas.to_csv('good_beta_seeds.csv', index=False)
print('\nSaved good beta seeds to good_beta_seeds.csv')
