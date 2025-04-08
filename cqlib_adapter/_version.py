# This code is part of cqlib.
#
# Copyright (C) 2025 China Telecom Quantum Group.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Version number"""
import os
import sys

if sys.version_info < (3, 10, 0):  # pragma: no cover
    raise SystemError(
        "cqlib requires Python 3.10 or higher. \n"
        "Please upgrade your Python version to continue using this library. \n"
        "You can download the latest version of Python at https://www.python.org/downloads/\n"
    )

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT_DIR, "VERSION.txt")) as version_file:
    VERSION = version_file.read().strip()

version: str
__version__: str

__version__ = version = VERSION
