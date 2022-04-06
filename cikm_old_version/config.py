import yaml

config_file = 'config_file/config_fb15k_subject1.yaml'

class ConfigDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def _make_config_dict(obj):
    if isinstance(obj, dict):
        return ConfigDict({k: _make_config_dict(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_make_config_dict(x) for x in obj]
    else:
        return obj

def config(configf=config_file):
    with open(configf, 'r') as f:
        _config_dict = _make_config_dict(yaml.load(f))
    return _config_dict
