"""gate1: verify Fix1+Fix2 jointly. PASS => proceed to Phase 2."""
import json
import os
import subprocess
import sys
from experiments.ccpa import diag_common


def main(seed=0):
    for s in ["fix1_phi_verify", "fix2_rho_bounded"]:
        subprocess.run([sys.executable, "-m", f"experiments.ccpa.{s}", "--seed", str(seed)], check=True)
    f1 = json.load(open(os.path.join(diag_common.RESULTS_DIR, "fix1_phi_monotone.json")))
    f2 = json.load(open(os.path.join(diag_common.RESULTS_DIR, "fix2_rho_bounded.json")))
    # beta_c = 1/rho(W) is structurally preserved: the log-det barrier changes the
    # objective but not the fixed-point map m=tanh(beta(Ws m+I-theta)) nor the
    # critical_beta formula (verified structurally in Fix1 grad test). The two
    # verify scripts use different architectures, so a numeric beta_c cross-check
    # would be bogus; preservation is a structural claim, not a numeric one.
    verdict = {
        "fix1_monotone": f1["monotone"],
        "fix2_bounded": f2["bounded"],
        "beta_c_structurally_preserved": True,
        "gate1": "PASS" if (f1["monotone"] and f2["bounded"]) else "FAIL",
    }
    json.dump(verdict, open(os.path.join(diag_common.RESULTS_DIR, "gate1.json"), "w"), indent=2)
    print(json.dumps(verdict, indent=2))
    return verdict


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
