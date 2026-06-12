#!/usr/bin/env python3
import sys
if sys.version_info < (3, 11):
    import typing
    from typing_extensions import NotRequired
    typing.NotRequired = NotRequired

exec(open("/data/home/2025030902017/seed_layer_output7/seed_layer.py").read())
