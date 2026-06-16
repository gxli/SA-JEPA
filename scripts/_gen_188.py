import glob, os

src = "/configs/experiments"
for f in sorted(glob.glob(f"{src}/gen_186_ngc_*.yaml")):
    with open(f) as fh:
        content = fh.read()
    content = content.replace("weight: 5.0", "weight: 10.0")
    dst = f.replace("gen_186_ngc", "gen_188_ngc")
    with open(dst, "w") as fh:
        fh.write(content)
    print(f"  {os.path.basename(f)} -> {os.path.basename(dst)}")
print("Done.")
