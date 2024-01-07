# util functions for RGAN
# Status: in developing

import json


def read_json(filename):
    with open(filename) as buf:
        return json.loads(buf.read())
