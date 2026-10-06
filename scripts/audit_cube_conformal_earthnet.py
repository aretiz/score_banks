#!/usr/bin/env python3
import numpy as np

from run_caps_z_earthnet_from_cache import (
    load_cache, filter_valid_fraction, subset,
    fit_pipeline, evaluate_scores,
)
from spt.innovation_filter import (
    MultiscaleInnovationFilter,
    template_box_scan,
)
from spt.caps_earthnet import all_raw_scores

SEED = 20260823
WINDOWS = [1,2,4,8,16]
ALPHA = 0.05


def split5(data):
    cube = np.unique(data["cube_id"])
    rng = np.random.default_rng(SEED)
    groups = np.array_split(rng.permutation(cube), 5)

    out = []
    for g in groups:
        idx = np.flatnonzero(np.isin(data["cube_id"], g))
        out.append(subset(data, idx))
    return out


def cp(ref, x):
    ref = np.sort(np.asarray(ref, float))
    x = np.asarray(x, float)
    ge = len(ref) - np.searchsorted(ref, x, side="left")
    return (1 + ge) / (len(ref) + 1)


def cube_max(score, cube):
    ids = np.unique(cube)
    vals = np.array([
        np.max(score[cube == g]) for g in ids
    ])
    return ids, vals


def precision_raw(data, filt, refs):
    s = template_box_scan(
        data["z"], data["valid"], filt, WINDOWS
    )
    e = np.column_stack([
        -np.log(cp(refs[k], s[k]))
        for k in sorted(refs)
    ])
    return e.max(1)


cal = filter_valid_fraction(
    load_cache("outputs/earthnet/caps_cal.npz", "nchw"), 0.05
)
test = filter_valid_fraction(
    load_cache("outputs/earthnet/caps_test.npz", "nchw"), 0.05
)

innovation_d, local_d, component_d, scene_d, audit_d = split5(cal)

print("cube counts:",
      *[len(np.unique(x["cube_id"])) for x in
        [innovation_d, local_d, component_d, scene_d, audit_d]],
      "test=", len(np.unique(test["cube_id"])))

filt = MultiscaleInnovationFilter(
    lags=(1,2,4,8,16),
    ridge=10.0,
    batch_size=32,
).fit(innovation_d)

local_cal, comp_norm, _ = fit_pipeline(
    local_d, component_d, scene_d, WINDOWS, 30
)

# CAPS raw window statistic -> cube maximum
scene_raw = all_raw_scores(
    scene_d["z"], scene_d["log_sigma"], scene_d["valid"],
    local_cal, comp_norm, WINDOWS
)
audit_raw = all_raw_scores(
    audit_d["z"], audit_d["log_sigma"], audit_d["valid"],
    local_cal, comp_norm, WINDOWS
)
test_raw = all_raw_scores(
    test["z"], test["log_sigma"], test["valid"],
    local_cal, comp_norm, WINDOWS
)

_, caps_ref = cube_max(scene_raw["caps_z"], scene_d["cube_id"])
_, caps_audit = cube_max(audit_raw["caps_z"], audit_d["cube_id"])
_, caps_test = cube_max(test_raw["caps_z"], test["cube_id"])

# Precision statistic -> cube maximum
comp = template_box_scan(
    component_d["z"], component_d["valid"], filt, WINDOWS
)
refs = {k: np.asarray(v, float) for k,v in comp.items()}

_, prec_ref = cube_max(
    precision_raw(scene_d, filt, refs),
    scene_d["cube_id"],
)
_, prec_audit = cube_max(
    precision_raw(audit_d, filt, refs),
    audit_d["cube_id"],
)
_, prec_test = cube_max(
    precision_raw(test, filt, refs),
    test["cube_id"],
)

for name, ref, audit, tst in [
    ("CAPS-Z", caps_ref, caps_audit, caps_test),
    ("Precision", prec_ref, prec_audit, prec_test),
]:
    pa = cp(ref, audit)
    pt = cp(ref, tst)

    print(f"\n{name}")
    print("scene_cal cubes =", len(ref))
    print("audit alarm =", np.mean(pa <= ALPHA))
    print("test alarm  =", np.mean(pt <= ALPHA))
