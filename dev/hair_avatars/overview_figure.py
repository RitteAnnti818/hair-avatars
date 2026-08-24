"""
HairAvatars method overview figure.
Panels:
  (a) Baseline: fully-rigid FLAME-triangle binding (GaussianAvatars)
  (b) Strand structure discovery (hair mask -> K=32 clustering -> root/tip order)
  (c) Strand Soft-Rigging (ours): static + dynamic offset, root-anchored chain accumulation
  (d) Strand-Coherence Regularization + qualitative result
All photo insets are REAL data generated earlier in this project (no synthetic images).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.cluster.vq import kmeans2
from PIL import Image
from flame_model.flame import FlameHead

np.random.seed(0)
PROJECT = "/hdd2/hee_data/GaussianAvatars"

# ---------------------------------------------------------------- shared data
flame = FlameHead(shape_params=300, expr_params=100)
verts = flame.v_template.detach().cpu().numpy()
faces = flame.faces.detach().cpu().numpy()
hair_fid = flame.mask.f.hair.detach().cpu().numpy()
centroids = verts[faces[hair_fid]].mean(axis=1)
K = 32
_, labels = kmeans2(centroids, K, minit='++', seed=0)

with open(f"{PROJECT}/dev/hair_avatars/phase1_strands_k32.json") as f:
    strands = json.load(f)

COL_BASE = "#b0b0b0"
COL_HAIR = "#8B5E3C"
COL_ROOT = "#2c7fb8"
COL_TIP = "#f03b20"
COL_PANEL_A = "#eef2f7"
COL_PANEL_B = "#fdf3e7"
COL_PANEL_C = "#eef8ee"
COL_PANEL_D = "#f6eefc"


def mesh_view(ax, hair_mask_faces=None, cluster_labels=None, elev=0, azim=90, highlight_face_ids=None, highlight_colors=None):
    tris = verts[faces]
    base_mask = np.ones(faces.shape[0], dtype=bool)
    if hair_mask_faces is not None:
        base_mask[hair_mask_faces] = False
    ax.add_collection3d(Poly3DCollection(tris[base_mask], facecolor=COL_BASE, edgecolor='none', alpha=1.0))
    if hair_mask_faces is not None and cluster_labels is None and highlight_face_ids is None:
        ax.add_collection3d(Poly3DCollection(tris[hair_mask_faces], facecolor=COL_HAIR, edgecolor='none', alpha=1.0))
    if cluster_labels is not None:
        cmap = plt.get_cmap('gist_ncar')
        colors = cmap(np.linspace(0, 1, K))
        for k in range(K):
            sel = hair_mask_faces[cluster_labels == k]
            if len(sel):
                ax.add_collection3d(Poly3DCollection(tris[sel], facecolor=colors[k], edgecolor='none', alpha=1.0))
    if highlight_face_ids is not None:
        face_colors = np.tile(np.array([0.93, 0.90, 0.88]), (faces.shape[0], 1))
        face_colors[hair_mask_faces] = np.array([0.93, 0.90, 0.88])
        for fids, col in zip(highlight_face_ids, highlight_colors):
            n = len(fids)
            grad = np.linspace(0.3, 1.0, n)
            for j, fid in enumerate(fids):
                face_colors[fid] = np.array(col) * grad[j]
        ax.add_collection3d(Poly3DCollection(tris, facecolor=face_colors, edgecolor='none', alpha=1.0))
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax.view_init(elev=elev, azim=azim)
    ax.axis('off')
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


# ============================================================== figure layout
fig = plt.figure(figsize=(20, 11))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1], hspace=0.28, wspace=0.18)

def panel_bg(ax_bbox_ax, color, label):
    rect = FancyBboxPatch((0.01, 0.02), 0.98, 0.96, transform=ax_bbox_ax.transAxes,
                           boxstyle="round,pad=0.01,rounding_size=0.02",
                           linewidth=1.2, edgecolor="#888888", facecolor=color, zorder=-10)
    ax_bbox_ax.add_patch(rect)
    ax_bbox_ax.text(0.03, 0.95, label, transform=ax_bbox_ax.transAxes, fontsize=15, fontweight='bold', va='top')


# ---------------------------------------------------------------- (a) baseline
axA = fig.add_subplot(gs[0, 0])
axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.axis('off')
panel_bg(axA, COL_PANEL_A, "(a) Baseline: fully-rigid binding")

for k, (tx, label) in enumerate([(0.28, "t"), (0.68, "t+1")]):
    tri = plt.Polygon([(tx-0.15, 0.55), (tx+0.15, 0.55), (tx, 0.78)], closed=True,
                       facecolor="#d8d8d8", edgecolor="black", zorder=2)
    axA.add_patch(tri)
    ell = mpatches.Ellipse((tx, 0.635), 0.11, 0.055, angle=15 if k == 0 else -10,
                            facecolor="#4472c4", edgecolor="black", zorder=3)
    axA.add_patch(ell)
    axA.text(tx, 0.42, f"frame {label}", ha='center', fontsize=11)
axA.annotate("", xy=(0.53, 0.66), xytext=(0.43, 0.66),
             arrowprops=dict(arrowstyle="->", lw=1.5))
axA.text(0.48, 0.70, "$R_i(t)$", ha='center', fontsize=11)
axA.text(0.5, 0.24, r"same relative pose on the triangle, every frame", ha='center', fontsize=10.5, style='italic')
axA.text(0.5, 0.155, r"$\mu'(t)=k_i(t)R_i(t)\mu+T_i(t)$", ha='center', fontsize=13)
axA.text(0.5, 0.045,
         "no independent freedom $\\Rightarrow$ can't represent\nthin, overlapping hair strands",
         ha='center', va='center', fontsize=9.5, color="#444444",
         bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#cccccc"))


# ---------------------------------------------------------------- (b) strand structure
axB0 = fig.add_subplot(gs[0, 1], projection='3d')
mesh_view(axB0, hair_mask_faces=hair_fid, elev=0, azim=90)
axB0.set_title("hair region\n(FLAME face mask, 656/10144 tri)", fontsize=10.5, y=0.95)

axB1 = fig.add_subplot(gs[0, 2], projection='3d')
mesh_view(axB1, hair_mask_faces=hair_fid, cluster_labels=labels, elev=0, azim=90)
axB1.set_title("K=32 strand clustering\n(k-means on triangle centroids)", fontsize=10.5, y=0.95)

axB2 = fig.add_subplot(gs[0, 3], projection='3d')
example_strands = [0, 8, 16, 24]
fids_list = [np.array(strands[str(k)]["face_ids"]) for k in example_strands]
cols_list = [(0.85, 0.1, 0.1), (0.1, 0.3, 0.85), (0.1, 0.6, 0.15), (0.85, 0.55, 0.0)]
mesh_view(axB2, hair_mask_faces=hair_fid, highlight_face_ids=fids_list, highlight_colors=cols_list, elev=0, azim=90)
axB2.set_title("root(dark)"+r"$\rightarrow$"+"tip(bright)\nordering per strand", fontsize=10.5, y=0.95)

# a shared translucent panel background behind the 3 mesh subplots
bgax = fig.add_axes([0.255, 0.51, 0.735, 0.46], zorder=-10)
bgax.axis('off')
panel_bg(bgax, COL_PANEL_B, "(b) Strand structure discovery")


# ---------------------------------------------------------------- (c) soft-rigging (ours)
axC = fig.add_subplot(gs[1, 0:2])
axC.set_xlim(0, 1); axC.set_ylim(0, 1); axC.axis('off')
panel_bg(axC, COL_PANEL_C, "(c) Strand Soft-Rigging (Ours)")

n_link = 6
xs = np.linspace(0.12, 0.88, n_link)
y0 = 0.62
radii = np.linspace(0.012, 0.05, n_link)
for j in range(n_link):
    w_j = j / (n_link - 1)
    color = np.array([0.17, 0.45, 0.72]) * (1 - w_j) + np.array([0.94, 0.23, 0.12]) * w_j
    if j > 0:
        # dashed circle showing growing freedom radius
        circ = Circle((xs[j], y0), radii[j] + 0.025, facecolor='none', edgecolor=color, ls='--', lw=1.3, alpha=0.8)
        axC.add_patch(circ)
    dot = Circle((xs[j], y0), 0.018, facecolor=color, edgecolor='black', zorder=5)
    axC.add_patch(dot)
    if j < n_link - 1:
        axC.annotate("", xy=(xs[j+1]-0.02, y0), xytext=(xs[j]+0.02, y0),
                     arrowprops=dict(arrowstyle="->", lw=1.2, color='gray'))
    tag = "root\n" + r"$\Delta_0=0$" if j == 0 else (r"tip" if j == n_link - 1 else f"$j={j}$")
    axC.text(xs[j], y0 + 0.11, tag, ha='center', fontsize=9.5)
    axC.text(xs[j], y0 - 0.10, f"$w_{{{j}}}={w_j:.1f}$", ha='center', fontsize=8.5, color='gray')

axC.text(0.5, 0.86, "scalp (fixed)"+r"$\longrightarrow$"+"strand tip (free)", ha='center', fontsize=11.5, fontweight='bold')

formula_c = (
    r"$\delta_j(t)=\bar\Delta_j+\varepsilon_j(t)$" + "     " + r"$w_j=\frac{j}{n_s-1}$" + "\n\n"
    r"$\Delta_j(t)=\sum_{k=0}^{j} w_k\,\delta_k(t)$" + "\n\n"
    r"$\mu'_j(t)=k_i(t)R_i(t)\mu_j+T_i(t)+R_i(t)\cdot\Delta_j(t)$"
)
axC.text(0.5, 0.30, formula_c, ha='center', va='center', fontsize=12.5,
         bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#888888"))
axC.text(0.02, 0.02, r"$\bar\Delta_j$: static (time-invariant) shape offset   |   $\varepsilon_j(t)$: dynamic per-frame residual",
         fontsize=9, style='italic', ha='left')


# ---------------------------------------------------------------- (d) coherence + result
axD = fig.add_subplot(gs[1, 2:4])
axD.set_xlim(0, 1); axD.set_ylim(0, 1); axD.axis('off')
panel_bg(axD, COL_PANEL_D, "(d) Strand-Coherence Regularization + Result")

# derive child-axes rectangles from axD's ACTUAL figure position (avoids hardcoded overlap)
dx0, dy0, dx1, dy1 = axD.get_position().extents
dw, dh = dx1 - dx0, dy1 - dy0

formula_d = (
    r"$L_{static}=\mathrm{mean}(\mathrm{relu}(\|\bar\Delta_j\|_2-\tau_{static}))$" + "\n\n"
    r"$L_{dynamic}=\mathrm{mean}(\mathrm{relu}(\|\varepsilon_j(t)\|_2-\tau_{dynamic}))$" + "\n\n"
    r"$\tau_{dynamic}\ll\tau_{static}$: prefer the" + "\n" + "static explanation (Occam's razor)"
)
axD.text(0.27, 0.66, formula_d, ha='center', va='center', fontsize=11,
         bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#888888"))

# mini relu-threshold plot, top-right quadrant of the panel
inset = fig.add_axes([dx0 + 0.57 * dw, dy0 + 0.54 * dh, 0.38 * dw, 0.26 * dh])
xr = np.linspace(0, 0.5, 200)
tau_s, tau_d = 0.3, 0.05
inset.plot(xr, np.maximum(xr - tau_s, 0), color=COL_ROOT, lw=2, label=r'static ($\tau{=}0.3$)')
inset.plot(xr, np.maximum(xr - tau_d, 0), color=COL_TIP, lw=2, label=r'dynamic ($\tau{=}0.05$)')
inset.set_xlabel(r'$\|\cdot\|_2$', fontsize=8)
inset.set_ylabel('penalty', fontsize=8)
inset.legend(fontsize=7, loc='upper left', frameon=False)
inset.tick_params(labelsize=6.5)
inset.set_title("threshold penalty shape", fontsize=8)

# real qualitative result crop (GT vs ours vs error), hair-region only — bottom row of the panel
def crop_top(img_path, frac=0.42):
    im = np.asarray(Image.open(img_path).convert("RGB"))
    h = im.shape[0]
    return im[: int(h * frac)]

gt = crop_top(f"{PROJECT}/output/hair_c1_check_306/test/ours_30000/gt/01360.png")
ours = crop_top(f"{PROJECT}/output/hair_c1_check_306/test/ours_30000/renders/01360.png")
err = np.abs(gt.astype(np.float32) - ours.astype(np.float32)).mean(axis=-1)

n_img = 3
img_w = 0.28 * dw
gap = (dw - n_img * img_w) / (n_img + 1)
for i, (img, title) in enumerate([(gt, "GT"), (ours, "Ours (30k it.)"), (err, "|GT-Ours|")]):
    a = fig.add_axes([dx0 + gap + i * (img_w + gap), dy0 + 0.05 * dh, img_w, 0.26 * dh])
    if img.ndim == 2:
        a.imshow(img, cmap='inferno', vmin=0, vmax=40)
    else:
        a.imshow(img)
    a.set_title(title, fontsize=9)
    a.axis('off')

fig.text(dx0 + dw / 2, dy0 + 0.005,
         "Real render from output/hair_c1_check_306 (iter 30k) — not a mockup.",
         ha='center', fontsize=8, style='italic')

fig.suptitle("HairAvatars: Strand Soft-Rigging for Dynamic Hair in Rigged 3D Gaussian Avatars",
             fontsize=17, fontweight='bold', y=0.995)

out_path = f"{PROJECT}/dev/hair_avatars/overview_figure.png"
plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
print("saved to", out_path)
