"""Synthetic metadata-only tests for neutral_wb_domain_plan. No files, media, targets or pixels."""
import copy
import unittest

import neutral_wb_domain_plan as N

SHA = "c" * 64


def profiles():
    return {"profiles": {
        "4080/6600": {"mode": 4080, "kelvin": 6600, "status": "candidate", "source_shifts_observed": [[20, -60]],
                      "target_ratios": {"R/G": {"num": 310, "den": 151, "float": 2.05}, "B/G": {"num": 455, "den": 302, "float": 1.5}}},
        "770/None": {"mode": 770, "kelvin": None, "status": "unvalidated", "reasons": ["family not in nonzero-shift profile allowlist"]}},
        "allowlist": ["4080/6600"], "min_scenes": 3, "min_sessions": 2}


def row(name="DSCF0001.RAF", role="fit", mode=0, kelvin=None, shift=(80, -100), cam=(440, 302, 828, 0), auto=(461, 302, 995, 302), **kw):
    r = {"name": name, "shoot": "Shoot 1", "role": role, "session": 3, "captureDynamicRange": 400, "raf_sha256": SHA,
         "source_wb_mode": mode, "source_kelvin": kelvin, "source_wb_fine_tune_raw": list(shift),
         "cam_mul": list(cam), "auto_rgbg": list(auto)}
    r.update(kw)
    return r


def kelvin(name="DSCF0002.RAF", role="fit", **kw):
    return row(name, role, mode=4080, kelvin=6600, shift=(20, -60), cam=(636, 302, 414, 0), **kw)


def zero(name="DSCF0003.RAF", role="fit", **kw):
    return row(name, role, mode=4080, kelvin=2500, shift=(0, 0), cam=(302, 325, 1137, 0), auto=(289, 302, 1044, 302), **kw)


class Kinds(unittest.TestCase):
    def test_three_kinds_keep_distinct_coefficient_semantics(self):
        w, p, g = (N.plan_row(r, profiles()) for r in (row(), kelvin(), zero()))
        self.assertEqual((w["kind"], w["contract"], w["declared_coeff_source"], w["declared_levels_rgbg"]),
                         ("W", "v2-linear-autowb", "fuji_f00d", [461, 302, 995, 302]))
        self.assertEqual((p["kind"], p["contract"], p["declared_coeff_source"], p["declared_levels_rgbg"]),
                         ("P", N.P.CONTRACT, "fit_profile", [620, 302, 455, 302]))
        self.assertEqual((g["kind"], g["contract"], g["declared_coeff_source"], g["declared_levels_rgbg"]),
                         ("G", "v2-linear-asshot", "cam_mul", [302, 325, 1137, 0]))  # own G325, fourth slot 0 kept
        self.assertTrue(all(r["export_eligible"] and r["requires_new_decode"] is not None for r in (w, p, g)))
        self.assertEqual(g["requires_new_decode"], False); self.assertEqual(g["reuse_source"], "existing-G-identity")
        self.assertTrue(w["requires_new_decode"] and p["requires_new_decode"])
        self.assertNotIn("path", str(w) + str(p) + str(g))

    def test_auto_white_priority_zero_shift_green_360(self):
        r = N.plan_row(row("DSCF0004.RAF", mode=1, shift=(0, 0), cam=(302, 360, 1360, 0), auto=(427, 302, 751, 302)), profiles())
        self.assertEqual((r["kind"], r["declared_levels_rgbg"]), ("G", [302, 360, 1360, 0]))

    def test_unsupported_presets_excluded_without_fallback(self):
        plan = [N.plan_row(row(f"DSCF{100 + i:04d}.RAF", mode=m, shift=s), profiles())
                for i, (m, s) in enumerate([(770, (-120, -80))] * 9 + [(512, (0, -20))] * 2 + [(256, (60, -100))])]
        self.assertEqual(len(plan), 12)
        for r in plan:
            self.assertTrue(r["excluded"]); self.assertIsNone(r["kind"]); self.assertFalse(r["export_eligible"])
            self.assertIsNone(r["declared_levels_rgbg"])
        self.assertEqual(N.fit_rows(plan), [])
        self.assertEqual(N.plan_row(kelvin(source_kelvin=5600), profiles())["excluded"], True)  # other Kelvin, nonzero shift


class Reuse(unittest.TestCase):
    def test_reuse_only_for_current_fit_and_matching_kind(self):
        qw, qp = {"DSCF0001.RAF", "DSCF0002.RAF"}, {"DSCF0002.RAF"}
        w = N.plan_row(row(), profiles(), qualified_w=qw, qualified_p=qp)
        p = N.plan_row(kelvin(), profiles(), qualified_w=qw, qualified_p=qp)
        self.assertEqual((w["requires_new_decode"], w["reuse_source"]), (False, "qualified-fit-W"))
        self.assertEqual((p["requires_new_decode"], p["reuse_source"]), (False, "qualified-fit-P"))
        cross = N.plan_row(kelvin("DSCF0009.RAF"), profiles(), qualified_w={"DSCF0009.RAF"})
        self.assertEqual((cross["requires_new_decode"], cross["reuse_source"]), (True, None))   # W name never reuses as P

    def test_development_never_enters_fit_export_even_if_named_in_reuse_set(self):
        dev = row("DSCF0700.RAF", role="development")
        plan = N.plan_rows([dev, row()], profiles(), qualified_w={"DSCF0700.RAF"})
        d = plan[0]
        self.assertEqual(d["kind"], "W")
        self.assertFalse(d["export_eligible"]); self.assertIsNone(d["requires_new_decode"]); self.assertIsNone(d["reuse_source"])
        self.assertEqual([r["name"] for r in N.fit_rows(plan)], ["DSCF0001.RAF"])
        s = N.summarize(plan)
        self.assertEqual((s["fit_export_rows"], s["fit_new_decodes"], s["fit_reused"]), (1, 1, 0))
        self.assertEqual(s["by_role_kind"], {"development/W": 1, "fit/W": 1})


class TargetIsolation(unittest.TestCase):
    def test_target_hints_dropped_and_cannot_affect_selection(self):
        base = N.plan_row(kelvin(), profiles())
        hinted = kelvin(target_rgb=[999, 302, 111], frozen_source_only_policy={"basis": "unsupported"},
                        tiff_sha256="d" * 64, target_gain=[2, 1, 2], kind="W", declared_levels_rgbg=[1, 1, 1, 1])
        h = N.plan_row(hinted, profiles())
        self.assertEqual(h, base)
        self.assertEqual(set(h["source"]), set(N.SOURCE_FIELDS))


class Refusals(unittest.TestCase):
    def bad(self, r, prof=None, **kw):
        with self.assertRaises(ValueError):
            N.plan_row(r, prof or profiles(), **kw)

    def test_invalid_rows(self):
        for change in ({"role": "monitor"}, {"role": "finalHeldOut"}, {"role": "K"}, {"role": "T8"}, {"role": "dr100Eval"},
                       {"raf_sha256": "C" * 64}, {"raf_sha256": "c" * 63}, {"name": "../DSCF0001.RAF"}, {"name": "DSCF001.RAF"},
                       {"shoot": "Shoot ../1"}, {"session": True}, {"session": -1}, {"captureDynamicRange": 300},
                       {"source_wb_mode": 2}, {"source_wb_mode": 0, "source_kelvin": 6600}, {"source_wb_fine_tune_raw": [10, 0]},
                       {"source_wb_fine_tune_raw": [200, 0]}, {"cam_mul": [440, 302, 828, 5]}, {"cam_mul": [0, 302, 828, 0]},
                       {"auto_rgbg": [461, 302, 995, 300]}, {"auto_rgbg": [461, True, 995, 302]}, {"cam_mul": "440,302,828,0"}):
            r = row(); r.update(change)
            self.bad(r)
        r = row(); del r["raf_sha256"]
        self.bad(r)
        self.bad(kelvin(source_kelvin=None))

    def test_unqualified_profile_pattern_refused_not_fallback(self):
        self.bad(row("DSCF0005.RAF", mode=4080, kelvin=6600, shift=(40, -20), cam=(636, 302, 414, 0)))

    def test_altered_or_extra_profiles_refused(self):
        p = profiles(); p["profiles"]["4080/6600"]["target_ratios"]["R/G"] = {"num": 311, "den": 151}
        self.bad(kelvin(), p)
        p = profiles(); p["profiles"]["770/None"]["status"] = "candidate"
        self.bad(kelvin(), p)
        p = profiles(); p["allowlist"] = ["4080/6600", "4080/5600"]
        self.bad(kelvin(), p)
        p = profiles(); p["profiles"]["4080/6600"]["source_shifts_observed"] = [[20, -60], [40, -20]]
        self.bad(kelvin(), p)
        self.bad(kelvin(), {"not": "profiles"})

    def test_plan_rows_refusals(self):
        with self.assertRaises(ValueError):
            N.plan_rows([row(), row()], profiles())
        for bad in (["DSCF0001.RAF"], {"bad name"}, {"DSCF0001.RAF", 1}):
            with self.assertRaises(ValueError):
                N.plan_rows([row()], profiles(), qualified_w=bad)

    def test_profile_constants_bound_to_codec(self):
        self.assertEqual(list(N.P.PROFILE_LEVELS_RGBG), [620, 302, 455, 302])
        self.assertEqual(N.P.PROFILE_MANIFEST_SHA256, "6f6dac1c4bdd619ae1daacac36581866b1b71b059fc970f23145b89ffbed8ffc")
        before = copy.deepcopy(profiles())
        N.validate_frozen_profiles(profiles())
        self.assertEqual(profiles(), before)


if __name__ == "__main__":
    unittest.main()
