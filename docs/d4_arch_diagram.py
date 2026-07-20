"""Render the VPSC d4 architecture diagram (Attention-is-All-You-Need style).

Cleaner layout: single left→right data flow, three expert rows strictly aligned,
router on top, combine on bottom, no crossing arrows. matplotlib, no browser.
Output: docs/d4_arch.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 9), dpi=170)
ax.set_xlim(0, 140); ax.set_ylim(0, 90); ax.axis("off")
ax.set_facecolor("white")

PAL = dict(emb="#FFE0B2", proj="#BBDEFB", exp=["#C8E6C9", "#B3E5FC", "#D1C4E9"],
           scan="#FFF59D", gate="#FFCDD2", comb="#E1BEE7", head="#FFCC80", io="#ECEFF1")

def box(x, y, w, h, fc, ec="#37474F", lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw, zorder=3))

def txt(x, y, s, sz=11, w="normal", col="#111", ha="center"):
    ax.text(x, y, s, fontsize=sz, fontweight=("bold" if w == "bold" else "normal"),
            color=col, ha=ha, va="center", zorder=5)

def arr(x1, y1, x2, y2, col="#455A64", ls="-", lw=1.8, mut=14):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=mut,
                                 color=col, lw=lw, linestyle=ls, zorder=2,
                                 shrinkA=0, shrinkB=0))

# ---- Title ----
ax.text(70, 86, "VPSC d4 — Temporal-Timescale Mixture-of-Experts SNN", fontsize=17,
        fontweight="bold", ha="center")
ax.text(70, 82.5, "fused gated-trace  ·  surpasses Transformer on catgirl-110k BPE (5.641 vs 5.845)",
        fontsize=11.5, color="#546E7A", ha="center", style="italic")

# ---- Left column: input pipeline (vertical, single flow) ----
box(8, 48, 14, 8, PAL["io"]);   txt(15, 52, "tokens [B,T]", 11, "bold")
box(8, 36, 14, 8, PAL["emb"]);  txt(15, 40, "Embedding", 11, "bold"); txt(15, 37.5, "+ LayerNorm", 9, col="#546E7A")
arr(15, 48, 15, 44.5)  # tokens -> embed

# embed -> MoE entry (horizontal to the MoE block)
arr(22, 40, 30, 40)

# ---- MoE container ----
ax.add_patch(FancyBboxPatch((30, 8), 78, 66, boxstyle="round,pad=0.02,rounding_size=1.0",
                            fc="#FAFAFA", ec="#90A4AE", lw=1.3, zorder=1))
txt(69, 71, "Temporal MoE   (n_experts = 3,  distinct decay bands)", 12.5, "bold", "#37474F")

# ---- Router (top, spanning the experts) ----
box(44, 60, 50, 9, PAL["gate"])
txt(69, 65.5, "Router", 12, "bold")
txt(69, 62.8, r"per-step  $\|\Delta x\|$  ,  $\|x\|$" + "   →  expert logits", 10, col="#37474F")

# ---- Three experts (strictly aligned rows) ----
exY = [49, 38, 27]           # y-center of each expert row
decay = ["short  d∈[.50,.80]", "mid  d∈[.65,.90]", "long  d∈[.80,.99]"]
for i in range(3):
    y = exY[i]
    # input projection
    box(33, y-4.5, 12, 9, PAL["exp"][i]); txt(39, y+1.8, "input proj", 9.5, "bold"); txt(39, y-0.8, "→ 4·state", 9, col="#546E7A")
    # events
    box(48, y-4.5, 12, 9, "#FFFFFF"); txt(54, y+1.8, "events", 9.5, "bold"); txt(54, y-0.8, "θ-threshold", 8.5, col="#546E7A"); txt(54, y-2.8, "surrogate", 8, col="#789")
    # affine scan (the SG27B fused kernel)
    box(63, y-4.5, 16, 9, PAL["scan"]); txt(71, y+2.2, "affine scan", 9.5, "bold"); txt(71, y-0.2, r"$z_t = d\,z_{t-1}+b_t$", 9.5, col="#37474F"); txt(71, y-2.8, "O(log T), hard-spike", 8, col="#789")
    # expert label + decay band (left of the row)
    txt(33, y+5.8, f"expert {i}", 10.5, "bold", "#37474F", ha="left")
    txt(63, y+5.8, decay[i], 9, w="normal", col="#546E7A", ha="left")
    # horizontal flow within expert
    arr(45, y, 48, y); arr(60, y, 63, y)
    # router -> this expert (dashed gate weight, straight down)
    arr(69, 60, 69, y+4.5, col="#E53935", ls=(0, (4, 3)), lw=1.3, mut=10)

# ---- expert outputs converge to combine (bottom) ----
# each expert scan output (right edge x=79) -> down to a bus at y=16, then into combine
for i in range(3):
    y = exY[i]
    arr(79, y, 82, y)                      # scan -> small hop right
    arr(82, y, 82, 16, col="#455A64", lw=1.4)  # down the bus (vertical, no crossing)
box(82, 11, 16, 9, PAL["comb"]); txt(90, 17, "soft combine", 10.5, "bold"); txt(90, 13.8, r"$\Sigma_e\, w_e \cdot seq_e$", 9.5, col="#37474F")
# bus enters combine from top
arr(90, 16, 90, 16)  # noop keep

# ---- right column: output -> head -> loss (vertical) ----
arr(98, 15.5, 106, 15.5)                       # combine -> output norm
box(106, 11, 14, 9, PAL["io"]); txt(113, 16.5, "output", 9.5); txt(113, 13.5, "norm + proj", 8.5, col="#546E7A")
arr(113, 20, 113, 25)                          # output -> head
box(106, 25, 14, 9, PAL["head"]); txt(113, 30.5, "LM head", 11, "bold"); txt(113, 27.5, "logits [B,T,V]", 8.5, col="#546E7A")
arr(113, 34, 113, 39)                          # head -> loss
box(106, 39, 14, 9, "#FFFFFF"); txt(113, 44.5, "next-token", 9.5, "bold"); txt(113, 41.5, "CE  (+F opt.)", 9.5, col="#37474F")

# ---- Key facts panel (far right) ----
ax.add_patch(FancyBboxPatch((124, 11), 15, 60, boxstyle="round,pad=0.02,rounding_size=0.8",
                            fc="#F5F5F5", ec="#CFD8DC", lw=1, zorder=1))
txt(131.5, 68, "Key facts", 11.5, "bold")
facts = [
    "• affine scan z_t =",
    "  d·z_{t-1} + b_t,",
    "  O(log T) depth,",
    "  hard-spike kept",
    "• experts differ only",
    "  in decay d → temporal",
    "  horizons, not weights",
    "• router = input",
    "  change-norm",
    "  (transition signal)",
    "• SG29, 3 seeds:",
    "  d4 5.641 <",
    "  Transformer 5.845",
    "  (catgirl-110k BPE,",
    "  15M tok, d=128)",
]
for i, f in enumerate(facts):
    txt(124.6, 64 - i * 3.5, f, 8.8, ha="left", col="#263238")

# ---- bottom caption ----
ax.text(70, 3.5, "data flow:  tokens → embed → [ router × 3 decay-specialist experts (affine scan) → soft combine ] → head → CE",
        fontsize=11, ha="center", style="italic", color="#37474F")

plt.tight_layout()
plt.savefig("/Users/united_pooh/PyProjects/vpsc/docs/d4_arch.png", bbox_inches="tight",
            facecolor="white", pad_inches=0.15)
print("saved docs/d4_arch.png")
