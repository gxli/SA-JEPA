import glob, os

src = "/configs/experiments"
dst = "/configs_bk"

count = 0
for g in range(172, 186):
    for f in sorted(glob.glob(f"{src}/gen_{g}_*.yaml")):
        name = os.path.basename(f)
        with open(f) as fh:
            content = fh.read()
        with open(f"{dst}/{name}", "w") as fh:
            fh.write(content)
        os.remove(f)
        count += 1
        if count % 10 == 0:
            print(f"  moved {count} files...")

print(f"Done. {count} files moved to {dst}")
