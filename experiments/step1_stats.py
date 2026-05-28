import pandas as pd

p2e = pd.read_parquet('data_intermediate/triples_policy_entity.parquet')
print('=== Step1 预处理统计 ===')
print(f'三元组总数: {len(p2e)}')
print(f'subject_type分布:')
print(p2e['subject_type'].value_counts().to_string())
print(f'object_type分布:')
print(p2e['object_type'].value_counts().to_string())
print(f'predicate分布:')
print(p2e['predicate'].value_counts().to_string())

ent = pd.read_parquet('data_intermediate/enterprises_filtered.parquet')
print(f'\n企业数: {len(ent)}')
print(f'企业列: {list(ent.columns)}')
if 'text_with_industry' in ent.columns:
    sample = ent['text_with_industry'].iloc[0][:120]
    print(f'带行业文本示例: {sample}')
if 'industry_major' in ent.columns:
    print(f'\n行业大类分布(top10):')
    print(ent['industry_major'].value_counts().head(10).to_string())

pol = pd.read_parquet('data_intermediate/policies_clean.parquet')
print(f'\n政策数: {len(pol)}')
