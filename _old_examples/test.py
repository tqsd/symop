import importlib

m = importlib.import_module("symop_proto.envelopes.spectral_filters.transfer")
print("module loaded from:", m.__file__)
print("has GaussianLowpass:", hasattr(m, "GaussianLowpass"))
print("dir contains GaussianLowpass:", "GaussianLowpass" in dir(m))
