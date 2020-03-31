import os
import ruamel.yaml

class ConfigMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = ConfigMember(value)
        return value

class Config(dict):
    yaml = ruamel.yaml.YAML()
    yaml.allow_duplicate_keys = True

    def __init__(self, file_path, logger):
        # print(os.getcwd())
        assert os.path.exists(file_path), "ERROR: Config File doesn't exist."
        with open(file_path, 'r') as f:
            self.member = self.yaml.load(f)
            f.close()
        self.logger = logger

    def __getattr__(self, name):
        if name not in self.member:
            if self.logger is None:
                print("Miss no name '%s' in config ", name)
            else:
                self.logger.error("Miss no name '%s' in config ", name)
            return False
        value = self.member[name]
        if isinstance(value, dict):
            value = ConfigMember(value)
        return value

    def isExist(self, name):
        if name in self.member:
            return True
        return False


    def write_yml(self):
        path = self.member['NET_INFO_PATH']
        with open(path, 'w+') as fp:
            self.yaml.dump(self.member, fp)