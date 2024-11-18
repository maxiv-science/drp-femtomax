import h5py

with h5py.File("data/scan-103327_andor3_balor.h5") as fh:
    print(fh["/entry/instrument/balor/data"].shape)
    with h5py.File("data/scan-103327_small.h5", "w") as fd:
        grp = fd.create_group("/entry/instrument/balor")
        grp.create_dataset(
            "data", data=fh["/entry/instrument/balor/data"][:2], compression="gzip"
        )
