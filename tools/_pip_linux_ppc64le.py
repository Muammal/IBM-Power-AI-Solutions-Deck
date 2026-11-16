"""Run pip with dependency markers evaluated for Linux on ppc64le (used by check_requirements.py).

`pip --platform` selects ppc64le wheels, but pip still evaluates markers such as `sys_platform == "win32"` for
the machine it runs on. On a Windows or macOS laptop that pulls in OS-specific dependencies (e.g. pywin32) that
do not exist for ppc64le and makes a valid Linux requirements file look broken. This shim patches the marker
environment, then hands over to pip unchanged.
"""
import sys

from pip._vendor.packaging import markers

_original = markers.default_environment


def _linux_ppc64le():
    env = _original()
    env.update(sys_platform="linux", platform_system="Linux", os_name="posix", platform_machine="ppc64le",
               platform_release="5.14.0", implementation_name="cpython", platform_python_implementation="CPython")
    return env


markers.default_environment = _linux_ppc64le

if __name__ == "__main__":
    from pip._internal.cli.main import main
    sys.exit(main())
