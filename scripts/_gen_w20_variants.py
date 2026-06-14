import glob, os

EXPERIMENTS = "/configs/experiments"

specs = [
    (166, 172, 20.0),
    (167, 173, 20.0),
    (168, 174, 20.0),
    (169, 175, 20.0),
    (170, 176, 20.0),
    (171, 177, 20.0),
]

for src_gen, dst_gen, w in specs:
    pattern = f"{EXPERIMENTS}/gen_{src_gen}_*.yaml"
    for src_path in sorted(glob.glob(pattern)):
        with open(src_path) as f:
            content = f.read()
        content = content.replace("weight: 2.0", f"weight: {w}")
        base = os.path.basename(src_path).replace(f"gen_{src_gen}", f"gen_{dst_gen}")
        if "_w20" not in base:
            base = base.replace(".yaml", "_w20.yaml")
        dst_path = os.path.join(EXPERIMENTS, base)
        with open(dst_path, "w") as f:
            f.write(content)
        print(f"  {os.path.basename(src_path)} -> {os.path.basename(dst_path)}")
    print(f"gen_{src_gen} -> gen_{dst_gen}: done")

print("Done.")
