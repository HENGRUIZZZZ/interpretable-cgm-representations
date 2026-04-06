"""
Run GlucoVector v6 experiments on the *new* dataset (P1_final_with_D4_DI.zip),
and save **all outputs** under a `New_`-prefixed results folder.
"""

from __future__ import annotations

import os
from datetime import datetime

from New_eval_trainD1D2_testD4 import eval_ckpt_on_d4, train_on_d1d2


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "New_data", "P1_final_with_D4_DI", "P1_final")
DEFAULT_OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v6")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    data_root = os.environ.get("NEW_CGM_PROJECT_OUTPUT", DEFAULT_DATA_ROOT)
    out_root = os.environ.get("NEW_RESULTS_ROOT", DEFAULT_OUT_ROOT)
    seed = int(os.environ.get("NEW_SEED", "21"))

    os.makedirs(out_root, exist_ok=True)

    # --------------------
    # Exp1: SSPG (train D1+D2, test D4)
    # --------------------
    exp1_dir = os.path.join(out_root, "New_exp1_sspg")
    os.makedirs(exp1_dir, exist_ok=True)
    exp1_train_dir = os.path.join(exp1_dir, f"train_D1D2_{_stamp()}")
    train_on_d1d2(
        cgm_project_output=data_root,
        results_dir=exp1_train_dir,
        seed=seed,
        lambda_sspg=0.1,
        lambda_di=0.0,
        num_epochs=int(os.environ.get("NEW_EXP1_EPOCHS", "100")),
    )
    exp1_ckpt = os.path.join(exp1_train_dir, "autoencoder_p1_full.pt")
    exp1_eval_dir = os.path.join(exp1_dir, "New_eval_on_D4")
    eval_ckpt_on_d4(
        cgm_project_output=data_root,
        ckpt_path=exp1_ckpt,
        out_dir=exp1_eval_dir,
        target="sspg",
    )

    # --------------------
    # Exp2: DI (train D1+D2, test D4)
    # --------------------
    exp2_dir = os.path.join(out_root, "New_exp2_di")
    os.makedirs(exp2_dir, exist_ok=True)
    exp2_train_dir = os.path.join(exp2_dir, f"train_D1D2_{_stamp()}")
    train_on_d1d2(
        cgm_project_output=data_root,
        results_dir=exp2_train_dir,
        seed=seed,
        lambda_sspg=0.0,
        lambda_di=0.05,
        num_epochs=int(os.environ.get("NEW_EXP2_EPOCHS", "100")),
    )
    exp2_ckpt = os.path.join(exp2_train_dir, "autoencoder_p1_full.pt")
    exp2_eval_dir = os.path.join(exp2_dir, "New_eval_on_D4")
    eval_ckpt_on_d4(
        cgm_project_output=data_root,
        ckpt_path=exp2_ckpt,
        out_dir=exp2_eval_dir,
        target="di",
    )

    print("Done: Exp1/Exp2 finished. (Exp3–Exp5 scripts will be run next.)")


if __name__ == "__main__":
    main()

