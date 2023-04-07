import importlib
from _collections_abc import Mapping, Sequence

def get_instance_from_conf(conf, *args, **kwargs):
    if isinstance(conf, (str, int, float, bool)):
        return conf
    params = conf.get("params", dict())
    _params = {}
    _params.update(kwargs)
    for k, v in params.items():
        if isinstance(v, Mapping) and "target" in v:
            v = get_instance_from_conf(conf=v)
        elif isinstance(v, Sequence) and not isinstance(v, str):
            v = [get_instance_from_conf(conf=i) for i in v]
        _params[k] = v
    return get_obj_from_str(conf["target"])(**_params)
    
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
