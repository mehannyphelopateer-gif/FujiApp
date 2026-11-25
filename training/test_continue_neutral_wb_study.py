"""One-shot continuation publication refuses unrelated paths/index; no real git or data access."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import continue_neutral_wb_study as C


class Publication(unittest.TestCase):
    def test_unrelated_index_refuses_before_any_write(self):
        with tempfile.TemporaryDirectory() as td,patch.object(C.D,'ROOT',Path(td)),patch.object(C,'git',return_value='other.py'),patch.object(C,'run') as run:
            p=Path(td)/'proof.json';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'staged'):C.publish([p],'message')
            run.assert_not_called()

    def test_exact_evidence_publication(self):
        with tempfile.TemporaryDirectory() as td,patch.object(C.D,'ROOT',Path(td)):
            p=Path(td)/'proof.json';p.write_text('{}');commands=[]
            with patch.object(C,'git',side_effect=['','proof.json']),patch.object(C,'run',side_effect=lambda args:commands.append(args)):
                C.publish([p],'message')
            self.assertEqual(commands,[['git','add','--','proof.json'],['git','commit','-m','message'],['git','push','origin','master']])

    def test_path_outside_repo_refused(self):
        with tempfile.TemporaryDirectory() as td,patch.object(C.D,'ROOT',Path(td)),patch.object(C,'run') as run:
            with self.assertRaises(ValueError):C.publish([Path('/outside/proof.json')],'message')
            run.assert_not_called()


if __name__=='__main__':unittest.main()
