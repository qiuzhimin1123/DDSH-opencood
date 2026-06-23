import spconv

try:
    import spconv.pytorch as spconv
except ImportError:
    pass


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        return out.replace_feature(new_features)
    out.features = new_features
    return out
