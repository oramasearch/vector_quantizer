import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the benchmark results
df = pd.read_csv('../benchmark_results.csv')

# Set up the plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create a figure with multiple subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Timing metrics
ax1.plot(df['n_samples'], df['fit_time_ms'], marker='o', label='Fit Time')
ax1.plot(df['n_samples'], df['compression_time_ms'], marker='o', label='Compression Time')
ax1.set_xlabel('Number of Samples')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Processing Time vs Dataset Size')
ax1.legend()
ax1.set_xscale('log')
ax1.set_yscale('log')

# Plot 2: Quality metrics
ax2.plot(df['n_samples'], df['reconstruction_error'], marker='o', label='Reconstruction Error')
ax2.plot(df['n_samples'], df['recall'], marker='o', label='Recall@10')
ax2.set_xlabel('Number of Samples')
ax2.set_ylabel('Score')
ax2.set_title('Quality Metrics vs Dataset Size')
ax2.legend()
ax2.set_xscale('log')

# Plot 3: Memory reduction
ax3.plot(df['n_samples'], (1 - df['memory_reduction_ratio']) * 100, marker='o')
ax3.set_xlabel('Number of Samples')
ax3.set_ylabel('Memory Reduction (%)')
ax3.set_title('Memory Reduction vs Dataset Size')
ax3.set_xscale('log')

# Plot 4: Time per sample
df['time_per_sample'] = (df['compression_time_ms']) / df['n_samples']
ax4.plot(df['n_samples'], df['time_per_sample'], marker='o')
ax4.set_xlabel('Number of Samples')
ax4.set_ylabel('Compression Time per Sample (ms)')
ax4.set_title('Scaling Efficiency')
ax4.set_xscale('log')
ax4.set_yscale('log')

plt.tight_layout()
plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
plt.close()