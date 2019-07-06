import re


class Config:
    def __init__(self, path):
        self.path = path
        self.load(path)

    def load(self, path=None):
        if path is None:
            path = self.path
        with open(path, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if '=' not in line:
                    continue  # skip blank lines
                key, value = line.split('=')
                self.addattr(key.strip(), value.strip())

    def addattr(self, key, val):
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

    def __str__(self):
        return "\n".join([f"{k}={v}" for k, v in self.__dict__.items()])

    def printdic(self):
        for k, v in self.__dict__.items():
            print([k, v])


if __name__ == '__main__':
    config = Config('testConfig.txt')
    config.load()
    # print(config)
    config.printdic()
