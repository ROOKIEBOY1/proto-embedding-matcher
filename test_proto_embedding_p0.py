"""Regression tests for evaluation integrity; no simulated embedding scores."""
import unittest
from decimal import Decimal
import proto_embedding_p0 as p


class IntegrityTests(unittest.TestCase):
    def test_parser_preserves_enum_identity_and_signed_ids(self):
        items = p.parse_proto('''// enum Fake { BAD = 9; }
        enum CameraModeType { A = 0; B = -1; }
        /* ignored = 7; */ enum WorkingModeType { C = 0; }''')
        self.assertEqual([(x.enum, x.name, x.number) for x in items],
                         [('CameraModeType', 'A', 0), ('CameraModeType', 'B', -1),
                          ('WorkingModeType', 'C', 0)])

    def test_numeric_validation_checks_full_enum(self):
        items = p.parse_proto('enum VideoFrameRateType { RATE_24FPS=1; RATE_48FPS=2; RATE_60FPS=3; RATE_120FPS=4; RATE_1440FPS=5; }')
        self.assertEqual(p.numeric_candidates('48 FPS', items), [1])
        self.assertEqual(p.numeric_candidates('144 FPS', items), [])
        self.assertEqual(p.numeric_candidates('48.0 FPS', items), [1])
        self.assertIsNone(p.numeric_candidates('Auto', items))

    def test_normalization_is_explicit_and_strategy_separated(self):
        item = p.Candidate('CameraModeType', 'CAMERA_MODE_TYPE_SLOWMOTION', 0)
        self.assertEqual(p.describe(item, 'A'), item.name)
        self.assertEqual(p.describe(item, 'B'), 'Camera Mode Type: Slowmotion')
        self.assertEqual(p.describe(item, 'C'), 'Camera Mode: Slow Motion')

    def test_threshold_uses_dev_only_and_balances_classes(self):
        t = p.calibrate([(0.9, True), (0.8, True), (0.6, False), (0.4, False)])
        self.assertGreater(t, 0.6)
        self.assertLessEqual(t, 0.8)

    def test_fixture_has_disjoint_queries_and_consistent_labels(self):
        import json
        rows = json.loads(p.DATA.read_text(encoding='utf-8'))
        self.assertGreaterEqual(sum(x['split'] == 'test' and x['expected'] is not None for x in rows), 100)
        dev = {x['field'] + ': ' + x['value'] for x in rows if x['split'] == 'dev'}
        test = {x['field'] + ': ' + x['value'] for x in rows if x['split'] == 'test'}
        self.assertFalse(dev & test)
        names = {(x.enum, x.name) for x in p.parse_proto(p.PROTO.read_text())}
        for row in rows:
            if row['expected']:
                self.assertIn((row['expected_enum'], row['expected']), names)
        self.assertEqual(len({x['id'] for x in rows}), len(rows))


if __name__ == '__main__':
    unittest.main()
