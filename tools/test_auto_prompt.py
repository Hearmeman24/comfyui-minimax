#!/usr/bin/env python3
"""Validate shipped auto-prompt graph wiring; optional live /object_info schema.

Run: python3 tools/test_auto_prompt.py [--schema /path/OpenRouterSimple.json]
No GPU generation or provider requests are made.
"""
import argparse
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = None
WIDGETS = ['model', 'reasoning_effort', 'timeout_seconds', 'temperature',
           'max_tokens', 'response_format', 'zdr', 'regenerate',
           'system_prompt', 'user_prompt']


def check_links(test, graph):
    nodes = {n['id']: n for n in graph['nodes']}
    links = {}
    for raw in graph.get('links', []):
        link = ([raw[k] for k in ('id', 'origin_id', 'origin_slot', 'target_id', 'target_slot', 'type')]
                if isinstance(raw, dict) else raw)
        ident, source, out_slot, target, in_slot, kind = link
        test.assertNotIn(ident, links)
        links[ident] = link
        if source == -10:
            test.assertIn(ident, graph['inputs'][out_slot]['linkIds'])
        else:
            test.assertIn(source, nodes)
            test.assertIn(ident, nodes[source]['outputs'][out_slot].get('links') or [])
        if target == -20:
            test.assertIn(ident, graph['outputs'][in_slot]['linkIds'])
        else:
            test.assertIn(target, nodes)
            test.assertEqual(nodes[target]['inputs'][in_slot]['link'], ident)
    for node in nodes.values():
        for index, socket in enumerate(node.get('inputs', [])):
            if socket.get('link') is not None:
                test.assertEqual(links[socket['link']][3:5], [node['id'], index])
        for index, socket in enumerate(node.get('outputs', [])):
            for ident in socket.get('links') or []:
                test.assertEqual(links[ident][1:3], [node['id'], index])


class AutoPromptContract(unittest.TestCase):
    def test_all_standalone_variants(self):
        paths = [p for p in (ROOT / 'workflows').rglob('*Auto Prompt*.json')
                 if 'I2V' in p.name or 'T2V' in p.name]
        self.assertEqual(len(paths), 4)
        for path in paths:
            with self.subTest(workflow=path.name):
                doc = json.loads(path.read_text())
                graphs = [doc, *doc.get('definitions', {}).get('subgraphs', [])]
                simple = [(g, n) for g in graphs for n in g['nodes'] if n['type'] == 'OpenRouterSimple']
                self.assertEqual(len(simple), 1)
                self.assertFalse(any(n['type'] == 'OpenRouterNode' for g in graphs for n in g['nodes']))
                graph, node = simple[0]
                for g in graphs:
                    check_links(self, g)
                vals = node['widgets_values_named']
                self.assertEqual(list(vals), WIDGETS)
                self.assertEqual(node['widgets_values'], list(vals.values()))
                self.assertEqual(vals['response_format'], 'text')
                self.assertTrue(vals['regenerate'])
                self.assertTrue(vals['system_prompt'])
                sockets = {s['name']: s for s in node['inputs']}
                expected_sockets = (['image', 'image_2', 'video', 'audio', 'user_prompt']
                                    if 'I2V' in path.name else ['image', 'video', 'audio', 'user_prompt'])
                self.assertEqual(list(sockets), expected_sockets)
                self.assertIsNotNone(sockets['user_prompt']['link'])
                self.assertEqual(sockets['image']['link'] is not None, 'I2V' in path.name)
                self.assertNotIn('api_key', sockets)
                self.assertNotIn('api_key', [s['name'] for s in graph['inputs']])
                self.assertEqual([o['name'] for o in node['outputs']], ['text', 'info', 'credits'])
                self.assertTrue(node['outputs'][0]['links'])
                for inst in doc['nodes']:
                    if inst['type'] == graph['id']:
                        self.assertEqual([s['name'] for s in inst['inputs']], [s['name'] for s in graph['inputs']])
                        self.assertNotIn('api_key', inst.get('widgets_values_named', {}))
                if SCHEMA:
                    required = SCHEMA['input']['required']
                    optional = SCHEMA['input']['optional']
                    self.assertEqual(WIDGETS, SCHEMA['input_order']['required'])
                    self.assertEqual([o['type'] for o in node['outputs']], SCHEMA['output'])
                    for key, val in vals.items():
                        spec = required[key]
                        if isinstance(spec[0], list):
                            self.assertIn(val, spec[0], key)
                        elif spec[0] in ('INT', 'FLOAT'):
                            self.assertGreaterEqual(val, spec[1]['min'])
                            self.assertLessEqual(val, spec[1]['max'])
                    for name, socket in sockets.items():
                        self.assertIn(name, {**required, **optional})
                        self.assertEqual(socket['type'], {**required, **optional}[name][0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--schema', type=Path)
    args = parser.parse_args()
    if args.schema:
        SCHEMA = json.loads(args.schema.read_text())
    unittest.main(argv=['test_auto_prompt.py'])
