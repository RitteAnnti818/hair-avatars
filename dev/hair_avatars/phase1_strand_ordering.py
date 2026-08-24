import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.cluster.vq import kmeans2
from flame_model.flame import FlameHead

np.random.seed(0)
K = 32

flame = FlameHead(shape_params=300, expr_params=100)
verts = flame.v_template.detach().cpu().numpy()
faces = flame.faces.detach().cpu().numpy()
hair_fid = flame.mask.f.hair.detach().cpu().numpy()

hair_faces = faces[hair_fid]
centroids = verts[hair_faces].mean(axis=1)

_, labels = kmeans2(centroids, K, minit='++', seed=0)

# crown reference point: topmost hair centroid (Y = up axis, following flame.py's teeth-construction convention)
crown_idx = np.argmax(centroids[:, 1])
crown_point = centroids[crown_idx]
print(f"crown point (top of head, hair region): {crown_point}")

# for each cluster, order member faces by distance from crown (root=near crown, tip=far/toward hairline)
strands = {}
for k in range(K):
    member_mask = labels == k
    member_fids = hair_fid[member_mask]
    member_centroids = centroids[member_mask]
    dists = np.linalg.norm(member_centroids - crown_point, axis=1)
    order = np.argsort(dists)
    ordered_fids = member_fids[order].tolist()
    strands[int(k)] = {
        "face_ids": ordered_fids,
        "root_to_tip_dist": dists[order].tolist(),
    }

out_json = "/hdd2/hee_data/GaussianAvatars/dev/hair_avatars/phase1_strands_k32.json"
with open(out_json, "w") as f:
    json.dump(strands, f, indent=1)
print("saved strand definitions to", out_json)

# visualize root->tip ordering as a color gradient (dark=root/near-crown, light=tip/far) for a few example strands
fig = plt.figure(figsize=(15, 5))
example_strands = [0, 8, 16, 24]

# build one combined per-face color array so matplotlib depth-sorts all triangles together
face_colors = np.tile(np.array([0.9, 0.87, 0.85]), (faces.shape[0], 1))
face_colors[hair_fid] = np.array([0.93, 0.9, 0.88])

cmap = plt.get_cmap('tab10')
for i, k in enumerate(example_strands):
    fids = np.array(strands[k]["face_ids"])
    n = len(fids)
    print(f"strand {k}: {n} faces")
    grad = np.linspace(0.25, 1.0, n)  # root(dark) -> tip(bright)
    base_color = np.array(cmap(i))[:3]
    for j, fid in enumerate(fids):
        face_colors[fid] = base_color * grad[j]

tris = verts[faces]
views = [(0, 90, "back/crown"), (90, -90, "top"), (0, 180, "other side")]
for vi, (elev, azim, name) in enumerate(views):
    ax = fig.add_subplot(1, len(views), vi + 1, projection='3d')
    ax.add_collection3d(Poly3DCollection(tris, facecolor=face_colors, edgecolor='none', alpha=1.0))
    ax.set_xlim(verts[:,0].min(), verts[:,0].max())
    ax.set_ylim(verts[:,1].min(), verts[:,1].max())
    ax.set_zlim(verts[:,2].min(), verts[:,2].max())
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(name)
    ax.axis('off')
    try:
        ax.set_box_aspect((1,1,1))
    except Exception:
        pass
fig.suptitle("Root(dark)->Tip(bright) ordering, 4 example strands (K=32)")

out_png = "/hdd2/hee_data/GaussianAvatars/dev/hair_avatars/phase1_strand_ordering.png"
plt.tight_layout()
plt.savefig(out_png, dpi=130)
print("saved to", out_png)
