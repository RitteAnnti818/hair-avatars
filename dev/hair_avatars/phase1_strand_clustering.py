import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.cluster.vq import kmeans2
from flame_model.flame import FlameHead

np.random.seed(0)

flame = FlameHead(shape_params=300, expr_params=100)
verts = flame.v_template.detach().cpu().numpy()   # (V, 3)
faces = flame.faces.detach().cpu().numpy()          # (F, 3)
hair_fid = flame.mask.f.hair.detach().cpu().numpy()

hair_faces = faces[hair_fid]                        # (Fh, 3)
centroids = verts[hair_faces].mean(axis=1)          # (Fh, 3) triangle centroids

print(f"hair faces: {len(hair_fid)}")

def plot_clusters(ax, labels, K, elev, azim):
    tris = verts[faces]
    other_mask = np.ones(faces.shape[0], dtype=bool)
    other_mask[hair_fid] = False
    other = Poly3DCollection(tris[other_mask], facecolor=(0.85, 0.8, 0.75), edgecolor='none', alpha=1.0)
    ax.add_collection3d(other)

    cmap = plt.get_cmap('tab20' if K <= 20 else 'gist_ncar')
    colors = cmap(np.linspace(0, 1, K))
    for k in range(K):
        sel = hair_fid[labels == k]
        if len(sel) == 0:
            continue
        coll = Poly3DCollection(tris[sel], facecolor=colors[k], edgecolor='none', alpha=1.0)
        ax.add_collection3d(coll)

    ax.set_xlim(verts[:,0].min(), verts[:,0].max())
    ax.set_ylim(verts[:,1].min(), verts[:,1].max())
    ax.set_zlim(verts[:,2].min(), verts[:,2].max())
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"K={K}")
    ax.axis('off')
    try:
        ax.set_box_aspect((1,1,1))
    except Exception:
        pass

Ks = [16, 32, 64]
fig = plt.figure(figsize=(15, 10))
for i, K in enumerate(Ks):
    centroid_out, labels = kmeans2(centroids, K, minit='++', seed=0)

    ax_back = fig.add_subplot(2, len(Ks), i + 1, projection='3d')
    plot_clusters(ax_back, labels, K, elev=0, azim=90)   # back/crown view (matches earlier "front view" panel which showed the cap)

    ax_top = fig.add_subplot(2, len(Ks), len(Ks) + i + 1, projection='3d')
    plot_clusters(ax_top, labels, K, elev=90, azim=-90)  # top-down view

    # cluster size stats
    sizes = np.bincount(labels, minlength=K)
    print(f"K={K}: cluster size min={sizes.min()} max={sizes.max()} mean={sizes.mean():.1f}")

plt.tight_layout()
out_path = "/hdd2/hee_data/GaussianAvatars/dev/hair_avatars/phase1_strand_clustering.png"
plt.savefig(out_path, dpi=120)
print("saved to", out_path)
