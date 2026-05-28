import json, h5py

meta_path = "/scratch/shahils/openpi/datasets/Libero-XVLA-format/libero_10_meta.json"
meta = json.load(open(meta_path))
p = meta["datalist"][0]
print("file:", p)
print("meta observation_key:", meta["observation_key"])

def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(name, obj.shape, obj.dtype)

with h5py.File(p, "r") as f:
    f.visititems(walk)
