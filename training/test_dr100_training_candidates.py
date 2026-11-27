"""Synthetic source-isolation/consistency checks; no RAFs or real metadata opened."""
import copy
import unittest
from unittest.mock import patch
import check_dr100_training_candidates as C


class CandidateGuards(unittest.TestCase):
    def test_exact_allowlist(self):
        self.assertEqual(len(C.candidate_paths(C.NAMES)),9)
        for names in (C.NAMES[:-1],(*C.NAMES[:-1],'DSCF1443.RAF'),('../escape.RAF',),tuple(reversed(C.NAMES))):
            with self.assertRaises(ValueError):C.candidate_paths(names)

    def test_fresh_metadata_complete_and_exact(self):
        expected={n:{'name':n,'model':'X100VI','path':p} for n,p in zip(C.NAMES,C.candidate_paths(C.NAMES))}
        fresh=list(copy.deepcopy(expected).values())
        self.assertEqual(C.check_fresh(expected,fresh),expected)
        for bad in (fresh[:-1],fresh+[fresh[0]],fresh[:-1]+[{'name':'DSCF1443.RAF'}]):
            with self.assertRaises(ValueError):C.check_fresh(expected,bad)
        fresh[0]['model']='different'
        with self.assertRaisesRegex(ValueError,'differs'):C.check_fresh(expected,fresh)

    def test_existing_output_stops_before_scope_or_source_read(self):
        with patch.object(C,'OUTPUT') as out,patch.object(C,'load_scope') as load:
            out.exists.return_value=True
            with self.assertRaisesRegex(ValueError,'overwrite'):C.run('a'*64)
            load.assert_not_called()

    def test_session_gap_rule_uses_adjacent_times(self):
        rows=[{'name':str(i),'dateTimeOriginal':t} for i,t in enumerate([
            '2026:11:16 19:00:00','2026:11:16 21:00:00','2026:11:16 23:01:00',
            '2026:11:17 02:01:00','2026:11:17 05:01:01'])]
        result=C.session_groups(rows)
        self.assertEqual([len(g['names']) for g in result],[4,1])


if __name__=='__main__':unittest.main()
