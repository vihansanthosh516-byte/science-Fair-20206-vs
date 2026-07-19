import json

# Check single KO TI
with open('output/single_ko_ti.json') as f:
    data = json.load(f)
print('Single KO TI count:', len(data))
for r in data:
    print(f'  {r["gene"]}: TI={r["therapeutic_index"]:.4f}, TumorC={r["tumor_collapse"]:.4f}, HealthyC={r["healthy_collapse"]:.4f}')

# Check dual KO TI
with open('output/dual_ko_ti.json') as f:
    data = json.load(f)
print('\nDual KO TI count:', len(data))
for r in data:
    print(f'  {r["gene_a"]}+{r["gene_b"]}: TI={r["therapeutic_index"]:.4f}, TumorC={r["tumor_collapse"]:.4f}, HealthyC={r["healthy_collapse"]:.4f}')