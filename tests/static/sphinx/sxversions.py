#!/usr/bin/python
from os.path import dirname, realpath

print(
    '{\n   "fake_addon" : "export PATH='
    + dirname(realpath(__file__))
    + '/bin:$PATH"\n}'
)
