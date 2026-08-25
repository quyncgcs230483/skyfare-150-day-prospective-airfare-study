"""Remove obsolete Error Analysis plots and repair remaining plot cells."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "artifacts/feature_research/notebooks/eda_03_error_analysis.ipynb"


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one occurrence, found {text.count(old)}: {old[:80]}")
    return text.replace(old, new)


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = {cell.get("id"): cell for cell in notebook["cells"]}

    set_source(
        cells["2ec11aa7"],
        """# EDA 03 Error Analysis: Old Absolute-Price Model

Failure-first audit of old `price_vnd` regression and `is_good_price`
classification. Main question: why offline metrics fail at first contact.

Active output is limited to six non-overlapping figures. Impossible
all-economic stress, duplicate plots, and deprecated classifier visuals are
removed. Source tables remain when they carry a reusable methodological lesson.
""",
    )

    setup = "".join(cells["62719cc8"]["source"])
    setup = setup.replace("    'all_economic_removed_stress': 'STRESS: all economic features removed',\n", "")
    start = setup.index("OLD_FIGURE_RENAMES = {")
    end = setup.index("def save_fig", start)
    setup = setup[:start] + setup[end:]
    setup = setup.replace("archive_existing_error_figures()\n", "")
    set_source(cells["62719cc8"], setup)

    set_source(
        cells["4d30beb6"],
        """## E2. Headline Failure Evidence: Offline Competence Degrades at First Contact

Offline evaluation contains same-schedule history. Honest cold-start removes
only five unavailable persistence features and keeps current-batch competitor
context. No impossible all-economic-removal lane is reported.
""",
    )

    helper = "".join(cells["47d54fe4"]["source"])
    helper = helper.replace(
        "    if mode == 'all_economic_removed_stress':\n        X[ECONOMIC_FEATURES] = np.nan\n        return X\n",
        "",
    )
    helper = helper.replace("serve_stress_pred_90 = predict_reg_for_mode(PRIMARY_MILESTONE, 'all_economic_removed_stress')\n", "")
    helper = helper.replace("serve_stress_r2 = reg_metrics(y90, serve_stress_pred_90)['r2']\n", "")
    stress_row = """    {
        'failure_mechanism': 'All economic features removed',
        'evidence': f"R2 {normal_r2:.3f} -> {serve_stress_r2:.3f}; competition also removed",
        'claim_scope': 'stress test only',
        'r2_drop': normal_r2 - serve_stress_r2,
    },
"""
    helper = replace_once(helper, stress_row, "")
    helper = helper.replace("    'stress test only': '#636363',\n", "")
    helper = helper.replace(
        "ax.set_title('Measured serving degradation with all-economic stress shown separately')",
        "ax.set_title('Cold-start damage dominates measured temporal drift')",
    )
    set_source(cells["47d54fe4"], helper)

    set_source(
        cells["77bbe568"],
        """### Main Failure Visuals F1 and F3

F1 shows prediction degradation. F3 identifies unavailable inputs. Distribution
collapse was removed because it duplicated F1 and relied on impossible stress.
""",
    )

    set_source(
        cells["394fe68e"],
        """# E2b. Main F1 and F3: visible serve-time failure
res = rf_results[PRIMARY_MILESTONE]
test_df = res['df_test']
y = res['y_reg']
offline_pred = res['reg_pred']
cold_start_pred = predict_reg_for_mode(PRIMARY_MILESTONE, 'cold_start_keep_cross_section')
offline_m = reg_metrics(y, offline_pred)
cold_start_m = reg_metrics(y, cold_start_pred)

sample_n = min(len(test_df), 70000)
rng = np.random.default_rng(42)
sample_idx = rng.choice(np.arange(len(test_df)), size=sample_n, replace=False)
lo = min(y.min(), offline_pred.min(), cold_start_pred.min())
hi = max(y.max(), offline_pred.max(), cold_start_pred.max())
pad = (hi - lo) * 0.03
lo, hi = lo - pad, hi + pad

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
panels = [
    (axes[0], offline_pred, f"OFFLINE - all features (R2={offline_m['r2']:.2f})", 'Blues'),
    (axes[1], cold_start_pred, f"AT SERVE - no same-schedule history\\ncompetitor context kept (R2={cold_start_m['r2']:.2f})", 'Oranges'),
]
for ax, pred, title, cmap in panels:
    ax.hexbin(y[sample_idx], pred[sample_idx], gridsize=60, bins='log', mincnt=1, cmap=cmap)
    ax.plot([lo, hi], [lo, hi], color='#333333', linestyle='--', linewidth=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel('Actual displayed price')
    ax.set_ylabel('Predicted price')
    use_million_axis(ax, 'both')
fig.suptitle('F1. Removing unavailable same-schedule history exposes cold-start loss')
save_fig(fig, 'main_F1_offline_vs_at_serve_pred_actual.png')
plt.show()

AVAILABILITY_FEATURES = SAME_FLIGHT_HISTORY_FEATURES + CROSS_SECTION_FEATURES
availability_base = df[AVAILABILITY_FEATURES + ['obs_index']].copy()
availability_base['contact'] = np.where(availability_base['obs_index'].eq(1), 'First contact', 'Later observations')
availability_rows = []
for contact, part in availability_base.groupby('contact'):
    for feature in AVAILABILITY_FEATURES:
        availability_rows.append({
            'contact': contact,
            'feature': feature,
            'feature_family': 'current-batch market' if feature in CROSS_SECTION_FEATURES else 'same-schedule history',
            'available_pct': part[feature].notna().mean() * 100,
        })
availability = pd.DataFrame(availability_rows)
availability.to_csv(TABLE_DIR / 'e_main_history_feature_availability.csv', index=False)
display(availability.pivot(index='feature', columns='contact', values='available_pct').reset_index())

fig, ax = plt.subplots(figsize=(11, 5))
sns.barplot(data=availability, y='feature', x='available_pct', hue='contact', ax=ax, order=AVAILABILITY_FEATURES)
ax.set_title('F3. Five persistence features disappear at first contact')
ax.set_xlabel('Rows where feature is available (%)')
ax.set_ylabel('')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.23), ncol=2, frameon=False)
save_fig(fig, 'main_F3_first_contact_feature_availability.png')
plt.show()
""",
    )

    set_source(
        cells["a43047c0"],
        """## E3. Fixability Test: Does a Serve-Time Database Rescue the Old Model?

Only causal pre-test proxies replace unavailable same-schedule history.
Current-batch competitor context remains available in every strategy.
""",
    )

    fix = "".join(cells["787ea719"]["source"])
    fix = fix.replace("    pred_stress = predict_reg_for_mode(milestone, 'all_economic_removed_stress')\n", "")
    fix = fix.replace("        ('all_economic_removed_stress', pred_stress, np.nan),\n", "")
    fix = fix.replace("    'all_economic_removed_stress',\n", "")
    fix = fix.replace("    'all_economic_removed_stress': 'STRESS\\nall econ removed',\n", "")
    fix = fix.replace("colors = ['#2B8CBE', '#F16913', '#636363', '#FD8D3C', '#FC9272', '#9E9AC8']", "colors = ['#2B8CBE', '#F16913', '#FD8D3C', '#FC9272', '#9E9AC8']")
    set_source(cells["787ea719"], fix)

    surface = "".join(cells["dc18d490"]["source"])
    set_source(cells["dc18d490"], surface.split("\nfig, axes =", 1)[0].rstrip() + "\n")
    residual = "".join(cells["78de89b4"]["source"])
    set_source(cells["78de89b4"], residual.split("\nfig, axes =", 1)[0].rstrip() + "\n")

    set_source(
        cells["2876a3cf"],
        """# F5. Direct OOT evidence without duplicate residual panels
if 'oot_metrics' not in globals():
    oot_artifacts = fit_june_oot_reference(PRIMARY_MILESTONE)
    oot_metrics = oot_artifacts['metrics']
    oot_metrics.to_csv(TABLE_DIR / 'e3_temporal_oot_retrain.csv', index=False)

display(oot_metrics)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.barplot(data=oot_metrics, x='eval_short', y='reg_r2', ax=axes[0], color='#0072B2')
axes[0].set_title('R2 remains above 0.76')
axes[0].set_xlabel('')
sns.barplot(data=oot_metrics, x='eval_short', y='reg_mape_pct', ax=axes[1], color='#E69F00')
axes[1].set_title('MAPE rises modestly')
axes[1].set_xlabel('')
fig.suptitle('F5. Temporal drift is real, but much smaller than cold-start loss')
save_fig(fig, 'main_F5_temporal_oot_drift.png')
plt.show()
""",
    )

    mechanism = "".join(cells["ef1fd5b7"]["source"])
    mechanism = mechanism.replace(
        "'feature_family': ['static' if c in STATIC_FEATURES else 'history_dependent' for c in FEATURES],",
        "'feature_family': [('same-schedule history' if c in SAME_FLIGHT_HISTORY_FEATURES else 'current-batch market' if c in CROSS_SECTION_FEATURES else 'static') for c in FEATURES],",
    )
    mechanism = mechanism.replace(
        "'metric': 'history_dependent_importance_sum',\n        'value': fi.loc[fi['feature_family'] == 'history_dependent', 'importance'].sum(),",
        "'metric': 'same_schedule_history_importance_sum',\n        'value': fi.loc[fi['feature_family'] == 'same-schedule history', 'importance'].sum(),",
    )
    mechanism = mechanism.split("\nfig, ax =", 1)[0].rstrip() + "\n" + """

fi90 = pd.read_csv(TABLE_DIR / 'e4_feature_importance_90days.csv')
fi90['corrected_family'] = np.select(
    [fi90['feature'].isin(SAME_FLIGHT_HISTORY_FEATURES), fi90['feature'].isin(CROSS_SECTION_FEATURES)],
    ['same-schedule history', 'current-batch market'],
    default='static',
)
fi90 = fi90.sort_values('importance')
colors = {'static': '#4C78A8', 'same-schedule history': '#E45756', 'current-batch market': '#72B7B2'}
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.barh(fi90['feature'], fi90['importance'], color=fi90['corrected_family'].map(colors))
ax.set_title('F6. Old RF relied mainly on warm-only persistence')
ax.set_xlabel('RF impurity importance (diagnostic only)')
ax.set_ylabel('')
fig.text(
    0.5, 0.01,
    '90days illustration only. price_lag_1step impurity importance varies 0.36-0.79 across 80/85/90days; F0 and F3 are primary evidence.',
    ha='center', fontsize=9, color='#555555'
)
fig.subplots_adjust(bottom=0.12)
save_fig(fig, 'main_F6_corrected_feature_importance_90days.png')
plt.show()
"""
    set_source(cells["ef1fd5b7"], mechanism)

    serve = "".join(cells["d2ce663f"]["source"])
    serve = serve.replace("['normal', 'cold_start_keep_cross_section', 'all_economic_removed_stress', 'proxy_fill_same_flight_history']", "['normal', 'cold_start_keep_cross_section', 'proxy_fill_same_flight_history']")
    set_source(cells["d2ce663f"], serve.split("\nfig, axes =", 1)[0].rstrip() + "\n")

    set_source(
        cells["bbde1956"],
        """## Appendix D. Deprecated Classifier Source Table

Keep label-regime and serve-skew tables only because they explain why static
`is_good_price` was replaced. No figure from deprecated target remains active.
""",
    )
    classifier = "".join(cells["0b195760"]["source"])
    classifier = classifier.replace("['normal', 'cold_start_keep_cross_section', 'all_economic_removed_stress', 'proxy_fill_same_flight_history']", "['normal', 'cold_start_keep_cross_section', 'proxy_fill_same_flight_history']")
    set_source(cells["0b195760"], classifier.split("\nfig, axes =", 1)[0].rstrip() + "\n")

    set_source(
        cells["b789ab0b"],
        """## E6. Failure Table to Requirements

Final table converts measured old-system failures into requirements. It keeps
honest cold-start, causal database proxy, temporal OOT, and deprecated-target
lessons separate. Impossible all-economic stress is removed.
""",
    )

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[PASS] curated {NOTEBOOK}")


if __name__ == "__main__":
    main()
