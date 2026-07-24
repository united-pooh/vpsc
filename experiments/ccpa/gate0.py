"""gate0: run all D-RC diagnostics and emit the Phase 0 verdict.

Rule (preregistered): >= 3 of 4 root causes (RC1/RC2/RC3/RC4) empirically
confirmed => PASS => proceed to Phase 1. < 3 => FAIL => STOP, record negative,
revisit the derivation. No post-hoc tuning.
"""
import json
import os
import subprocess
import sys
from experiments.ccpa import diag_common

RC_SCRIPTS = {"RC1": "d_rc1_components", "RC2": "d_rc2_errorfloor",
              "RC3": "d_rc3_rho_degeneracy", "RC4": "d_rc4_hessian_jacobian"}


def _confirm_RC1(j):
    fs = [r["F"] for r in j["rows"]]
    return (max(fs) - min(fs)) > 0.1 * abs(sum(fs) / len(fs))  # F non-monotone across beta


def _confirm_RC2(j):  # orthogonal (continuous) floor grows with beta
    fo = j["orthogonal"]
    return fo[-1] > 1.5 * fo[0]


def _confirm_RC3(j):  # rho exceeds rho_max without cap
    return j["rhos"][-1] > 0.9


def _confirm_RC4(j):  # rho(DG) approaches 1 at beta_c (Curie signal)
    bc = j["beta_c"]
    rows = j["rows"]
    near = min(rows, key=lambda r: abs(r["beta"] - bc))
    return near["rho_DG"] > 0.8


CHECKS = {"RC1": _confirm_RC1, "RC2": _confirm_RC2, "RC3": _confirm_RC3, "RC4": _confirm_RC4}


def main(seed=0):
    for rc, script in RC_SCRIPTS.items():
        subprocess.run([sys.executable, "-m", f"experiments.ccpa.{script}", "--seed", str(seed)], check=True)
    verdict = {}
    for rc, script in RC_SCRIPTS.items():
        with open(os.path.join(diag_common.RESULTS_DIR, script + ".json")) as f:
            j = json.load(f)
        verdict[rc] = {"confirmed": bool(CHECKS[rc](j))}
    n_conf = sum(v["confirmed"] for v in verdict.values())
    verdict["gate0"] = "PASS" if n_conf >= 3 else "FAIL"
    payload = {"seed": seed, "verdict": verdict, "n_confirmed": n_conf,
               "rule": ">=3 of 4 RCs confirmed => PASS"}
    path = os.path.join(diag_common.RESULTS_DIR, "gate0.json")
    json.dump(payload, open(path, "w"), indent=2)
    print(json.dumps(payload, indent=2))
    return verdict


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
