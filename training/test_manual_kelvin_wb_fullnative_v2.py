"""Synthetic contract checks: the transport revision cannot move a scientific goalpost."""
import copy
import unittest
from unittest.mock import patch

import qualify_manual_kelvin_wb_fullnative_v2 as V


class TransportContractTests(unittest.TestCase):
    def test_only_transport_fields_may_change(self):
        old={k:{'fixed':k} for k in ('scenes','base_options','primitive_limits','selection','sampling','limits','interpretation')}
        new=dict(old,parent_fullnative_manifest_sha256=V.PARENT_SHA,transport_revision=2,output_dir='new')
        V.unchanged_measurement(new,old)
        for field in old:
            changed=copy.deepcopy(new);changed[field]={'altered':True}
            with self.assertRaises(ValueError):V.unchanged_measurement(changed,old)
        for key,val in [('transport_revision',3),('parent_fullnative_manifest_sha256','0'*64)]:
            changed=dict(new);changed[key]=val
            with self.assertRaises(ValueError):V.unchanged_measurement(changed,old)

    def test_measurement_is_the_frozen_full_native_function(self):
        import qualify_manual_kelvin_wb_fullnative as F
        self.assertIs(V.F.full_statistics,F.full_statistics)

    def test_manifest_must_be_pinned_and_committed(self):
        with self.assertRaises(ValueError):V.load(None)
        with self.assertRaises(ValueError):V.load('0'*64)


if __name__=='__main__':unittest.main()
