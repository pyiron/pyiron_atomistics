__author__ = "Christoph Freysoldt"
__copyright__ = (
    "Copyright 2022, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Christoph Freysoldt"
__email__ = "freysoldt@mpie.de"
__status__ = "development"
__date__ = "Oct 15, 2022"

from threading import Lock
import json
import warnings
import os.path
from sys import executable as python_interpreter
import subprocess

from pyiron_base import Settings

_sxversion_lock = Lock()
_sxversions = None


def sxversions(refresh=False):
    """Get available SPHInX versions
    returns: a dict
    """
    global _sxversions
    if not refresh and _sxversions is dict:
        return _sxversions

    def do_update(warn_what, newfile):
        if isinstance(sxv, dict):
            for v in sxv.keys():
                if v in version_origin.keys() and _sxversions[v] != sxv[v]:
                    warnings.warn(
                        "Overriding sxversion '{}' from {} with the one from {}".format(
                            v, version_origin[v], newfile
                        )
                    )
                version_origin[v] = newfile
            _sxversions.update(sxv)
        else:
            warnings.warn("Failed to parse " + warn_what + newfile)

    _sxversion_lock.acquire()
    try:
        if refresh:
            _sxversions = None
        if _sxversions is None:
            _sxversions = dict()
            version_origin = dict()
            for p in Settings().resource_paths:
                jsonfile = os.path.join(p, "sphinx", "sxversions.json")
                if os.path.exists(jsonfile):
                    with open(jsonfile) as f:
                        sxv = json.load(f)
                    do_update("", jsonfile)

                jsonscript = os.path.join(p, "sphinx", "sxversions.py")
                if os.path.exists(jsonscript):
                    proc = subprocess.run(
                        [python_interpreter, jsonscript],
                        text=True,
                        stdout=subprocess.PIPE,
                        cwd=os.path.join(p, "sphinx"),
                    )
                    if proc.returncode != 0:
                        warnings.warn(
                            jsonscript
                            + " failed with exitcode "
                            + str(proc.returncode)
                            + ": \n"
                            + proc.stdout
                            + proc.stderr
                        )
                    else:
                        try:
                            sxv = json.loads(proc.stdout)
                            do_update("output from ", jsonscript)
                        except json.decoder.JSONDecodeError as ex:
                            raise RuntimeError(
                                "json decoder error from "
                                + jsonscript
                                + " output:\n"
                                + proc.stdout
                            ) from ex
    finally:
        _sxversion_lock.release()
    return _sxversions
