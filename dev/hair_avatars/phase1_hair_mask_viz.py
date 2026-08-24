import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from flame_model.flame import FlameHead

flame = FlameHead(shape_params=300, expr_params=100)
verts = flame.v_template.detach().cpu().numpy()  # (V, 3)
faces = flame.faces.detach().cpu().numpy()        # (F, 3)
hair_fid = flame.mask.f.hair.detach().cpu().numpy()

hair_mask = np.zeros(faces.shape[0], dtype=bool)
hair_mask[hair_fid] = True

print(f"verts: {verts.shape}, faces: {faces.shape}, hair faces: {hair_mask.sum()}")

def plot_view(ax, elev, azim, title):
    tris = verts[faces]  # (F, 3, 3)
    other = Poly3DCollection(tris[~hair_mask], facecolor=(0.8, 0.75, 0.7), edgecolor='none', alpha=1.0)
    hair = Poly3DCollection(tris[hair_mask], facecolor='red', edgecolor='none', alpha=1.0)
    ax.add_collection3d(other)
    ax.add_collection3d(hair)
    ax.set_xlim(verts[:,0].min(), verts[:,0].max())
    ax.set_ylim(verts[:,1].min(), verts[:,1].max())
    ax.set_zlim(verts[:,2].min(), verts[:,2].max())
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.axis('off')
    try:
        ax.set_box_aspect((1,1,1))
    except Exception:
        pass

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')
plot_view(ax1, elev=0, azim=90, title="Front view")
plot_view(ax2, elev=0, azim=0, title="Side view")
plot_view(ax3, elev=90, azim=-90, title="Top view")

out_path = "/tmp/claude-1017/-hdd2-hee-data/5e5a7a2c-9b99-4620-877d-64f6ead246e3/scratchpad/phase1_hair_mask_viz.png"
plt.tight_layout()
plt.savefig(out_path, dpi=120)
print("saved to", out_path)
