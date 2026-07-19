import json
with open('output/single_ko_ti.json') as f:
    data = json.load(f)
print('Single KO TI count:', len(data))
for r in data[:5]:
    print(f'  {r["gene"]}: TI={r["therapeutic_index"]:.2f}')

with open('output/dual_ko_ti.json') as f:
    data = json.load(f)
print('Dual KO TI count:', len(data))
for r in data[:5]:
    print(f'  {r["gene_a"]}+{r["gene_b"]}: TI={r["therapeutic_index"]:.2f}')