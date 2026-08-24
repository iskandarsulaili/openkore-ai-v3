#!/usr/bin/env python3
"""Regenerate OpenKore .dist walkability maps from the live client GAT files.

.dist format (OpenKore): 'V#' + u16 ver + u16 w + u16 h + w*h bytes (1=walkable, 0xFF=unreachable).
GAT v2 format: 'GRAT' + u16 ver + u32 w + u32 h + w*h cells of 20 bytes (u32 tile_type + 4 floats).
Tile type bit 0x01 = walkable.

Usage: python3 gen_dist.py <gat_dir> <out_dir> [map1 map2 ...]
  gat_dir: dir containing *.gat (e.g. /home/lot399/Ragnarok/client/data)
  out_dir: dir to write *.dist (e.g. /home/lot399/openkore-ai-v3/fields)
  maps: optional list of map names (without .gat); default = all *.gat
"""
import os, struct, sys, collections

def gen_dist(gat_path, out_path):
    d = open(gat_path, 'rb').read()
    if d[0:4] != b'GRAT':
        return False, f"bad magic {d[0:4]!r}"
    ver = struct.unpack('<H', d[4:6])[0]
    w, h = struct.unpack('<II', d[6:14])
    cell = 20
    if len(d) != 14 + w * h * cell:
        return False, f"size mismatch {len(d)} vs {14+w*h*cell}"
    # walkable = low byte of tile_type in {0,2,3,4,6} (rAthena map_gat2cell).
    # GAT v2 cell = 20 bytes: 4x float height (offset 0..15) + u32 type (offset 16).
    WALKABLE = {0, 2, 3, 4, 6}
    walk = bytearray(w * h)
    for i in range(w * h):
        tile_type = struct.unpack('<I', d[14 + i * cell + 16 : 14 + i * cell + 20])[0]
        walk[i] = 1 if (tile_type & 0xFF) in WALKABLE else 0
    # BFS distance from all walkable cells; unreachable = 0xFF
    dist = bytearray(w * h)
    dist[:] = b'\xff' * (w * h)
    q = collections.deque()
    for i in range(w * h):
        if walk[i]:
            dist[i] = 1
            q.append(i)
    while q:
        i = q.popleft()
        x, y = i % w, i // w
        for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
            if 0 <= nx < w and 0 <= ny < h:
                j = ny * w + nx
                if walk[j] and dist[j] == 0xFF:
                    dist[j] = dist[i] + 1
                    q.append(j)
    out = b'V#' + struct.pack('<HHH', 4, w, h) + bytes(dist)
    open(out_path, 'wb').write(out)
    return True, f"{w}x{h} walkable={sum(walk)}/{w*h}"

def main():
    gat_dir, out_dir = sys.argv[1], sys.argv[2]
    maps = sys.argv[3:] or [f[:-4] for f in os.listdir(gat_dir) if f.endswith('.gat')]
    os.makedirs(out_dir, exist_ok=True)
    ok = fail = 0
    for m in sorted(maps):
        gp = os.path.join(gat_dir, m + '.gat')
        if not os.path.exists(gp):
            print(f"SKIP {m}: no {gp}")
            continue
        good, msg = gen_dist(gp, os.path.join(out_dir, m + '.dist'))
        if good:
            ok += 1
            print(f"OK {m}: {msg}")
        else:
            fail += 1
            print(f"FAIL {m}: {msg}")
    print(f"\n{ok} generated, {fail} failed")

if __name__ == '__main__':
    main()
