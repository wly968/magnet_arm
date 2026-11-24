import re, sys
p = r"scene_magnet/saves/train_log.txt"
rows=[]; stages=[]
for line in open(p, encoding="utf-8"):
    line=line.strip()
    if not line or line.startswith("#"): continue
    if "STAGE" in line:
        stages.append(line); continue
    d={}
    toks=line.split()
    d["step"]=int(toks[0]); d["stage"]=toks[1]
    for t in toks[2:]:
        if "=" in t:
            k,v=t.split("=",1); d[k]=v
    rows.append(d)
print("数据行:", len(rows), "| 升档行:", len(stages))
for s in stages: print("  ", s)
# 按档汇总
from collections import OrderedDict
by=OrderedDict()
for r in rows:
    by.setdefault(r["stage"], []).append(r)
def num(x):
    try: return float(str(x).rstrip("%"))
    except: return None
for st, rs in by.items():
    def series(k):
        return [num(r.get(k)) for r in rs if num(r.get(k)) is not None]
    relpos=series("relpos"); allp=series("all"); mrel=series("mrel"); eplen=series("eplen"); probe=series("probe"); pos=series("pos")
    def fmt(v): return "-" if not v else f"{v[0]:.2f} -> {v[-1]:.2f} (max {max(v):.2f})"
    print(f"\n{st}: {len(rs)} 行, step {rs[0]['step']} ~ {rs[-1]['step']}")
    if pos: print(f"   pos%    : {fmt(pos)}")
    if relpos: print(f"   relpos% : {fmt(relpos)}")
    if allp: print(f"   all%    : {fmt(allp)}")
    if mrel: print(f"   mrel    : {fmt(mrel)}")
    if eplen: print(f"   eplen   : {fmt(eplen)}")
    if probe: print(f"   probe%  : {fmt(probe)}")
