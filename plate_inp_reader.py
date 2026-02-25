import re
from typing import Dict, List, Tuple, Any
import numpy as np


def parse_inp_text(text: str) -> Dict[str, Any]:
    """Parse important blocks from an Abaqus .inp file using regular expressions.

    Returns a dictionary with keys like 'nodes' and 'elements'.
    """
    sections = {}

    # Find all star-directives and their following non-star lines
    pattern = re.compile(r'(?m)^(\*[^\n]*?)\n(?:(?!^\*).*(?:\n|$))*', re.MULTILINE)
    # Alternate, more robust approach: iterate lines and collect blocks
    lines = text.splitlines()
    header = None
    body_lines: List[str] = []
    for ln in lines:
        if ln.strip().startswith('*'):
            # flush previous
            if header is not None:
                sections.setdefault(header, []).append('\n'.join(body_lines))
            header = ln.strip()
            body_lines = []
        else:
            body_lines.append(ln)
    if header is not None:
        sections.setdefault(header, []).append('\n'.join(body_lines))

    result: Dict[str, Any] = {'nodes': {}, 'elements': []}

    # process node blocks
    for h, bodies in sections.items():
        hlow = h.lower()
        if hlow.startswith('*node'):
            for body in bodies:
                for line in body.splitlines():
                    line = line.strip()
                    if not line or line.startswith('**'):
                        continue
                    m = re.match(r"^\s*(\d+)\s*,\s*([+-]?[0-9.eE+-]+)\s*,\s*([+-]?[0-9.eE+-]+)\s*,\s*([+-]?[0-9.eE+-]+)", line)
                    if m:
                        nid = int(m.group(1))
                        x = float(m.group(2))
                        y = float(m.group(3))
                        z = float(m.group(4))
                        result['nodes'][nid] = (x, y, z)
    # process element blocks
    for h, bodies in sections.items():
        hlow = h.lower()
        if hlow.startswith('*element'):
            # try to capture element type from header
            mtype = None
            mt = re.search(r'type\s*=\s*([^,\s]+)', h, re.I)
            if mt:
                mtype = mt.group(1)
            for body in bodies:
                for line in body.splitlines():
                    line = line.strip()
                    if not line or line.startswith('**'):
                        continue
                    # split by commas and parse ints
                    parts = [p.strip() for p in line.split(',') if p.strip()]
                    if len(parts) >= 2:
                        try:
                            eid = int(parts[0])
                            conn = [int(p) for p in parts[1:]]
                            result['elements'].append({'id': eid, 'type': mtype, 'conn': conn})
                        except ValueError:
                            # skip malformed lines
                            continue

    return result


def parse_inp_file(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return parse_inp_text(text)


def nodes_to_array(nodes_dict: Dict[int, Tuple[float, float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert nodes dict to (ids array, coords Nx3 array) sorted by id."""
    ids = sorted(nodes_dict.keys())
    coords = np.array([nodes_dict[i] for i in ids], dtype=float)
    return np.array(ids, dtype=int), coords


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse an Abaqus .inp file (simple regex-based).')
    parser.add_argument('inp', nargs='?', default='Plate.inp')
    args = parser.parse_args()

    data = parse_inp_file(args.inp)
    n_nodes = len(data['nodes'])
    n_elems = len(data['elements'])
    print(f'Parsed {n_nodes} nodes and {n_elems} elements from {args.inp!s}')
    # print a few samples
    if n_nodes:
        ids, coords = nodes_to_array(data['nodes'])
        print('First 5 node ids and coords:')
        for i in range(min(5, ids.size)):
            print(f'  {ids[i]:5d}: {coords[i,0]:.6g}, {coords[i,1]:.6g}, {coords[i,2]:.6g}')
    if n_elems:
        print('First 5 elements:')
        for e in data['elements'][:5]:
            print(f"  id={e['id']:d}, type={e['type']}, conn={e['conn']}")
