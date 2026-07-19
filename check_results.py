import json

with open('output/dual_ko_results.json') as f:
    data = json.load(f)

# Sort by bliss_synergy
sorted_data = sorted(data, key=lambda x: x.get('bliss_synergy', 0), reverse=True)

print("Top 5 dual KOs by Bliss synergy:")
for i, r in enumerate(sorted_data[:5], 1):
    print(f'  {i}. {r["gene_a"]} + {r["gene_b"]}: C={r.get("collapse_score", 0):.4f}, Bliss={r.get("bliss_synergy", 0):.4f}')

print()
with open('output/single_ko_results.json') as f:
    data = json.load(f)
sorted_single = sorted(data, key=lambda x: x['collapse_score'], reverse=True)
print('Top 5 single KOs by collapse score:')
for i, r in enumerate(sorted_single[:5], 1):
    print(f'  {i}. {r["gene"]}: C={r["collapse_score"]:.4f}')