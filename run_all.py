"""
run_all.py
==========
End-to-end orchestrator. Run with:

    python run_all.py                 # full run (10 repeats)
    TB_QUICK=1 python run_all.py      # fast smoke test
    TB_DATA=/path/to/data.csv python run_all.py

Produces, under outputs/:
    tables/   all result tables as CSV
    figures/  all figures as 600-dpi PNG

Pipeline order mirrors the flow diagram:
  1 data prep -> 2 two-track model comparison -> select top-2 by Track A kappa
  -> 3 downstream analyses for both models on BOTH tracks -> 4 figures.
"""
from __future__ import annotations
import pandas as pd

import config as C
import data_prep
import evaluation as EV
import feature_analysis as FA
import cooccurrence as CO
import plotting as PL


def main():
    print("=" * 70)
    print(f"Tiger beetle classification re-run  (QUICK={C.QUICK}, "
          f"repeats={C.N_REPEATS})")
    print("=" * 70)

    # ---- 1. data -------------------------------------------------------
    d = data_prep.get_data()
    X, y, site_id, df = d["X"], d["y"], d["site_id"], d["df"]
    print("Cleaning:", d["cleaning_summary"])
    print("Sites   :", d["site_summary"])
    print("\nPer-class de-duplication (count lost per class):")
    print(d["dedup_report"])

    # ---- 2. model comparison, both tracks ------------------------------
    print("\n[2] Comparing all models on Track A and Track B ...")
    results, best_params_df = EV.compare_all_models(X, y, site_id)
    results.to_csv(C.TABLE_DIR / "model_comparison_full.csv", index=False)
    # best hyperparameters chosen in every repeat (for Table 3 / reproducibility)
    best_params_df.to_csv(
        C.TABLE_DIR / "best_hyperparameters_per_repeat.csv", index=False)

    def _fmt(track):
        sub = results[results["track"] == track].set_index("model")
        cols = {
            "test_accuracy_mean": "test_acc", "test_accuracy_std": "test_acc_sd",
            "test_f1_macro_mean": "test_f1", "test_f1_macro_std": "test_f1_sd",
            "test_kappa_mean": "test_kappa", "test_kappa_std": "test_kappa_sd",
            "cv_accuracy_mean": "cv_acc", "cv_f1_macro_mean": "cv_f1",
            "cv_kappa_mean": "cv_kappa",
            "train_ms_mean": "train_ms", "infer_ms_per_sample_mean": "infer_ms",
            "best_hyperparameters": "best_hyperparameters",
        }
        keep = [c for c in cols if c in sub.columns]
        return sub[keep].rename(columns=cols)

    _fmt("A").to_csv(C.TABLE_DIR / "table3_trackA_specimen_test.csv")
    _fmt("B").to_csv(C.TABLE_DIR / "table3B_trackB_unseen_site_test.csv")
    print("Track A (specimen-level held-out test):")
    print(_fmt("A").round(3))
    print("\nTrack B (location-independent / unseen-site test):")
    print(_fmt("B").round(3))

    # ---- select top-2 by Track A held-out kappa ------------------------
    top2 = EV.select_top2(results, track="A", metric="test_kappa_mean")
    print(f"\n>>> Top-2 models by Track A held-out kappa: {top2}")

    # ---- 3. downstream analyses for both models on BOTH tracks ---------
    cooc = CO.co_occurrence(df, site_id)
    cooc.to_csv(C.TABLE_DIR / "table8_cooccurrence.csv")

    downstream_params = []
    for track in ("A", "B"):
        print(f"\n[3] Downstream analyses - Track {track} ...")
        single_by_model, scheme_by_model, block_by_model = {}, {}, {}
        ablation_by_model, scatter_by_model, pearson_by_model = {}, {}, {}

        for m in top2:
            print(f"    model {m} ...", flush=True)
            best_params = FA.tune_once(m, X, y, site_id, track)
            downstream_params.append({
                "track": track, "model": m,
                **{k.replace("clf__", ""): v for k, v in best_params.items()}})
            oof, folds = FA.fit_folds(m, best_params, X, y, site_id, track)

            pct = FA.per_class_table(y, oof)
            pct.to_csv(C.TABLE_DIR / f"table7_perclass_{track}_{m}.csv")

            single_by_model[m] = FA.single_feature_importance(folds, X, y)
            scheme_by_model[m] = FA.permutation_schemes(folds, X, y)
            block_by_model[m] = FA.block_importance(folds, X, y)
            ablation_by_model[m] = FA.ablation(m, best_params, X, y, site_id, track)
            ablation_by_model[m].to_csv(
                C.TABLE_DIR / f"ablation_{track}_{m}.csv", index=False)

            # co-occurrence correlation using this model's per-class F1
            f1_series = pct["f1"]
            sh, pear, scat = CO.correlation(f1_series, cooc)
            sh.to_csv(C.TABLE_DIR / f"table5_shapiro_{track}_{m}.csv", index=False)
            pear.to_csv(C.TABLE_DIR / f"table6_pearson_{track}_{m}.csv", index=False)
            scatter_by_model[m] = scat
            pearson_by_model[m] = float(pear["pearson_r"].iloc[0])

        # ---- 4. figures for this track --------------------------------
        PL.two_panel_barh(
            single_by_model, "feature",
            "Mean decrease in Macro-F1 across CV folds",
            f"Cross-fold permutation-importance stability  (Track {track})",
            C.FIG_DIR / f"fig3_single_importance_track{track}.png")

        PL.two_panel_vbar(
            scheme_by_model, "scheme", "mean", "std",
            "Mean decrease in Macro-F1 across 5 CV folds",
            f"Permutation schemes compared  (Track {track})",
            C.FIG_DIR / f"fig4_perm_schemes_track{track}.png")

        PL.two_panel_barh(
            block_by_model, "feature",
            "Mean decrease in Macro-F1 across 5 CV folds (group permuted jointly)",
            f"Grouped (block) permutation importance  (Track {track})",
            C.FIG_DIR / f"fig5_block_importance_track{track}.png")

        PL.two_panel_vbar(
            ablation_by_model, "feature_set", "f1_macro_mean", "f1_macro_std",
            "Macro-F1 (5-fold CV)",
            f"Feature-group ablation  (Track {track})",
            C.FIG_DIR / f"ablation_track{track}.png")

        PL.two_panel_scatter(
            scatter_by_model, pearson_by_model,
            C.FIG_DIR / f"fig6_f1_vs_cooccurrence_track{track}.png")

    # best hyperparameters used for the downstream (figure) models
    pd.DataFrame(downstream_params).to_csv(
        C.TABLE_DIR / "best_hyperparameters_downstream.csv", index=False)

    # ---- Fig 7 : variance decomposition (model-agnostic, once) --------
    wf, vp, meta = CO.variance_decomposition(df, site_id)
    PL.variance_figure(wf, vp, meta, C.FIG_DIR / "fig7_variance_decomposition.png")
    wf.to_csv(C.TABLE_DIR / "fig7_within_site_fraction.csv")
    vp.to_csv(C.TABLE_DIR / "fig7_variance_partition.csv")

    print("\nDone. See outputs/tables and outputs/figures.")


if __name__ == "__main__":
    main()
