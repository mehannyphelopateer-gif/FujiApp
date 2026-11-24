"""No-media refusal tests: foreign paths/protected roles fail before any file read."""
import unittest
from pathlib import Path
from unittest.mock import patch
import verify_neutral_wb_fit_export_v2 as A


class ScopeTests(unittest.TestCase):
    def test_foreign_artifact_refused_before_read(self):
        s={'role':'fit','kind':'W','shoot':'Shoot 1','reuse':None}
        r={'file':'training/checkpoints/provia-input-v2-train-g/Shoot 999/input.fjlrg',
           'strips_file':'training/checkpoints/neutral-wb-fit-export/Shoot 1/C-native-strips.u16le'}
        with patch.object(Path,'read_bytes',side_effect=RuntimeError('unapproved read')) as read:
            with self.assertRaisesRegex(AssertionError,'allowlist'):A.artifact(s,r,'C','0'*64)
            read.assert_not_called()

    def test_protected_roles_refused_before_read(self):
        for role in ('development','monitor','finalHeldOut','K','T8','dr100Eval'):
            with self.subTest(role=role),patch.object(Path,'read_bytes',side_effect=RuntimeError('unapproved read')) as read:
                with self.assertRaises(AssertionError):A.artifact({'role':role},{},'C','0'*64)
                with self.assertRaises(AssertionError):A.check_scene({'role':role},{},'0'*64)
                read.assert_not_called()

    def test_namespace_escape_refused(self):
        for p in ('/etc/passwd','calibration-input/Shoot 1/xraw-phase3.tiff','training/checkpoints/neutral-wb-fit-export/../../other'):
            with self.subTest(path=p),self.assertRaises(AssertionError):A.inside(p)


if __name__=='__main__':unittest.main()
