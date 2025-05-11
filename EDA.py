import pandas as pd

# Load your dataset
df = pd.read_csv('/work/pi_wenlongzhao_umass_edu/25/data/routerbench/routerbench_0shot_test.csv')

# Step 1: Count and get top 5 domains
top_5_domains = df['eval_name'].value_counts().nlargest(5).index.tolist()

# Step 2: Save each top domain as a separate file
for domain in top_5_domains:
    domain_df = df[df['eval_name'] == domain]
    file_name = f"/work/pi_wenlongzhao_umass_edu/25/data/routerbench/routerbench_0shot_test_domainwise/{domain}.csv"
    domain_df.to_csv(file_name, index=False)
    print(f"✅ Saved {len(domain_df)} rows to {file_name}")
