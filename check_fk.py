import json

with open('output/fk_metrics.json') as f:
    m = json.load(f)

print('FK Metrics:')
print(f'  Analytical wave speed: {m["analytical_wave_speed"]:.4f} px/step')
print(f'  Numerical wave speed: {m["numerical_wave_speed"]:.4f} px/step')
print(f'  Error: {m["speed_error_percent"]:.1f}%')
print(f'  Clinical velocity: {m["wave_speed_um_per_hr"]:.2f} µm/hr')