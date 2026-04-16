


import h5py
import numpy as np


def dict_to_hdf5(d: dict, h5_path: str, group_name: str = "root", mode: str = "w"):
    """
    Save a (possibly nested) Python dict to an HDF5 file.

    Supported leaf types:
      - numpy arrays
      - numbers (int/float/bool)
      - strings
      - bytes
      - lists/tuples of numbers/strings (converted to numpy arrays)

    Nested dicts become groups.
    """
    def _write_item(g, key, value):
        key = str(key)

        if isinstance(value, dict):
            subg = g.require_group(key)
            for k2, v2 in value.items():
                _write_item(subg, k2, v2)

        elif isinstance(value, (np.ndarray,)):
            g.create_dataset(key, data=value)

        elif isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
            g.create_dataset(key, data=value)

        elif isinstance(value, (str, bytes)):
            # Store strings robustly
            dt = h5py.string_dtype(encoding="utf-8") if isinstance(value, str) else None
            g.create_dataset(key, data=value, dtype=dt)

        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value)
            if arr.dtype.kind in ("U", "S", "O"):
                # list of strings / mixed: store as variable-length UTF-8 strings where possible
                if all(isinstance(x, str) for x in value):
                    dt = h5py.string_dtype(encoding="utf-8")
                    g.create_dataset(key, data=np.array(value, dtype=object), dtype=dt)
                else:
                    raise TypeError(f"List for key='{key}' contains non-string objects; not supported.")
            else:
                g.create_dataset(key, data=arr)

        elif value is None:
            # HDF5 can't store None directly; store as an attribute marker
            ds = g.create_dataset(key, data=np.array([], dtype=np.uint8))
            ds.attrs["_is_none"] = True

        else:
            raise TypeError(f"Unsupported type for key='{key}': {type(value)}")

    with h5py.File(h5_path, mode) as f:
        root = f.require_group(group_name)
        # If overwriting within an existing file/group, clear the group first
        for k in list(root.keys()):
            del root[k]
        for k, v in d.items():
            _write_item(root, k, v)


def hdf5_to_dict(h5_path: str, group_name: str = "root") -> dict:
    """
    Load an HDF5 file (or group) back into a nested Python dict.
    """
    def _read_group(g):
        out = {}
        for key, item in g.items():
            if isinstance(item, h5py.Group):
                out[key] = _read_group(item)
            else:
                # dataset
                if item.attrs.get("_is_none", False):
                    out[key] = None
                else:
                    val = item[()]
                    # decode bytes -> str when appropriate
                    if isinstance(val, (bytes, np.bytes_)):
                        val = val.decode("utf-8")
                    # convert 0-d numpy arrays to python scalars
                    if isinstance(val, np.ndarray) and val.shape == ():
                        val = val.item()
                    out[key] = val
        return out

    with h5py.File(h5_path, "r") as f:
        g = f[group_name]
        return _read_group(g)


        