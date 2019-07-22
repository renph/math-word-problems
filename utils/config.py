import re

import json
import copy


class Config:
    def __init__(self, path=''):
        if path:
            self.load(path)

    @classmethod
    def from_dict(cls, dic):
        config = cls()
        config.__dict__.update(dic)

    @classmethod
    def from_json(cls, js):
        return cls.from_dict(json.loads(js))

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json(self):
        return json.dumps(self.__dict__)

    def load(self, path):
        with open(path, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if '=' not in line:
                    continue  # skip blank lines
                key, value = line.split('=')
                self.add(key.strip(), value.strip())

    def add(self, key, val):
        # print('cur %s  , %s'%(key,val))
        if not isinstance(val, str):
            value = val
        else:
            if val == 'True':
                value = True
            elif val == 'False':
                value = False
            elif val.startswith(r"'"):
                value = val.strip("' ")  # strip space and single quote
            else:
                num_format = re.compile(r"^\-?[0-9]*\.?[0-9]*$")
                isnumber = re.match(num_format, val)
                if isnumber:
                    if '.' in val:
                        value = float(val)
                    else:
                        value = int(val)
                else:
                    raise Exception(str([key, val, 'match failed']))
        self.__dict__[key] = value

    def __getitem__(self, item):
        return self.__dict__[item]

    def __str__(self):
        return "\n".join([f"{k}={v}" for k, v in self.__dict__.items()])
