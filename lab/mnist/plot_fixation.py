"""Composite fixation plot: accuracy & confidence vs fixation timestep,
poisson vs static, on one figure. Reads the two fixation_*.json files."""

import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(mode):
    path = os.path.join(RESULTS_DIR, f"fixation_{mode}.json")
    with open(path) as f:
        return json.load(f)


def main():
    import matplotlib.pyplot as plt

    pois = load("poisson")
    sta = load("static")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    # ---- accuracy ----
    ax[0].plot(pois["acc_t"], "-o", ms=3, color="#C44E52", label=f"poisson (peak {max(pois['acc_t']):.4f})")
    ax[0].plot(sta["acc_t"], "-o", ms=3, color="#4C72B0", label=f"static (peak {max(sta['acc_t']):.4f})")
    ax[0].axhline(0.9795, color="gray", ls="--", lw=0.8, label="MLP baseline 0.9795")
    ax[0].set_xlabel("fixation timestep")
    ax[0].set_ylabel("test accuracy")
    ax[0].set_title("Accuracy over prolonged fixation")
    ax[0].legend(loc="lower right")
    ax[0].set_ylim(0.5, 1.0)

    # ---- confidence ----
    ax[1].plot(pois["conf_t"], "-o", ms=3, color="#C44E52", label="poisson")
    ax[1].plot(sta["conf_t"], "-o", ms=3, color="#4C72B0", label="static")
    ax[1].set_xlabel("fixation timestep")
    ax[1].set_ylabel("mean confidence (softmax max)")
    ax[1].set_title("Confidence over prolonged fixation")
    ax[1].legend(loc="lower right")
    ax[1].set_ylim(0, 1.0)

    fig.suptitle("VPSC on MNIST: staring at one image — accuracy & confidence grow with fixation time",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fixation_composite.png")
    fig.savefig(out, dpi=120)
    print(f"[written {out}]")

    # ---- summary table ----
    print("\n=== fixation summary ===")
    print(f"{'mode':<10} {'acc@0':>8} {'acc@10':>8} {'acc@25':>8} {'peak':>8} {'peak_t':>8} {'conf@0':>8} {'conf@peak':>10}")
    for name, d in [("poisson", pois), ("static", sta)]:
        pt = d["acc_t"].index(max(d["acc_t"]))
        print(f"{name:<10} {d['acc_t'][0]:>8.4f} {d['acc_t'][10]:>8.4f} {d['acc_t'][25]:>8.4f} "
              f"{max(d['acc_t']):>8.4f} {pt:>8d} {d['conf_t'][0]:>8.4f} {d['conf_t'][pt]:>10.4f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
