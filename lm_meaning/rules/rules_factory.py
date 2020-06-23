import os
import sys
from pkgutil import iter_modules

from setuptools import find_packages


class RuleFactory:
    def __init__(self):
        pass

    def find_rule(self, path, callange_to_find):
        modules = list()
        for pkg in [''] + find_packages(path):
            pkgpath = path + '/' + pkg.replace('.', '/')
            if sys.version_info.major == 2 or (sys.version_info.major == 3 and sys.version_info.minor < 6):
                for _, name, ispkg in iter_modules([pkgpath]):
                    if not ispkg:
                        modules.append(pkg + '.' + name)
            else:
                for info in iter_modules([pkgpath]):
                    if not info.ispkg:
                        modules.append(pkg + '.' + info.name)

        found_instructions = [module for module in modules if module.find('.' + callange_to_find) > -1]
        if len(found_instructions) > 0:
            found_instructions = found_instructions[0]
            if found_instructions.startswith('.'):
                found_instructions = found_instructions[1:]
        else:
            found_instructions = None

        return found_instructions

    def get_rule(self, challenge_name):
        module_name = self.find_rule(os.path.dirname(os.path.abspath(__file__)) + '/trex',
                                     challenge_name)
        try:
            mod = __import__('trex.' + module_name, fromlist=[challenge_name])
        except:
            assert (ValueError('rule name not found!'))

        return getattr(mod, challenge_name)()
