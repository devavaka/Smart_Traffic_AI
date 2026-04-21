import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Fix for VSCode/script display
import matplotlib.pyplot as plt
import os

H, W = 450, 800
heatmap = np.zeros((H, W), dtype=np.float32)

# Simulate 60 frames of vehicles at random-ish positions
np.random.seed(0)
# Two congested zones + random scatter
zone1 = [(200 + int(np.random.normal(0,20)), 150 + int(np.random.normal(0,15))) for _ in range(30)]
zone2 = [(580 + int(np.random.normal(0,30)), 300 + int(np.random.normal(0,20))) for _ in range(30)]
all_positions = zone1 + zone2

snapshots = []
os.makedirs('heatmaps', exist_ok=True)
for frame_i, (cx, cy) in enumerate(all_positions):
    cx = max(20, min(W-20, cx))
    cy = max(20, min(H-20, cy))
    
    # Draw circle + GaussianBlur (like production heatmap.py)
    tmp = np.zeros_like(heatmap)
    cv2.circle(tmp, (cx, cy), 20, 1.0, -1)
    tmp = cv2.GaussianBlur(tmp, (41, 41), 0)  # radius*2+1 |1
    heatmap += tmp
    heatmap *= 0.997  # Match production decay
    
    if frame_i in [9, 29, 59]:
        snapshots.append((frame_i+1, heatmap.copy()))

# Create output dir
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for i, (ax, (frame_n, snap)) in enumerate(zip(axes, snapshots)):
    norm = cv2.normalize(snap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    ax.imshow(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
    ax.set_title(f'After {frame_n} frames', fontweight='bold')
    ax.axis('off')
    
    # Save individual PNGs
    png_path = f'heatmaps/snapshot_{frame_n:02d}.png'
    cv2.imwrite(png_path, colored)
    print(f'Saved: {png_path}')

plt.suptitle('TECHNIQUE 4: Heatmap Accumulation + Decay (COLORMAP_JET) - FIXED', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.ion()  # Interactive mode
plt.show()
plt.pause(0.1)  # Ensure display

print('Red/yellow = high activity zones | Blue = low activity | Decay factor = 0.997/frame')
print('PNG snapshots saved in heatmaps/ folder. GUI is running separately.')

