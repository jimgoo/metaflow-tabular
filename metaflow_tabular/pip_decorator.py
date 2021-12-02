import functools
import logging
import signal
from importlib import import_module
from typing import Dict

logger = logging.getLogger(__name__)


def pip(libraries: Dict[str, str]):
    """
    A Flow decorator which mimics @conda, except installs pip deps. Use @conda instead whenever possible.

    Note: this requires 3rd party modules to be imported _inside_ the flow/step this decorator scopes; otherwise you
    will get ModuleNotFound errors. Also note that this decorator has to be on the line before @conda is used.

    To install wheels from a specific source url, put the url after the library name separated by a pipe, i.e.
    @pip({'torch|your.urlr/here':1.8.1)

    Will check to see if the pkg is already installed before re-installing. This means that this will not install the
    exact pinned version if the library already exists.

    Based on: https://github.com/Netflix/metaflow/issues/24#issuecomment-571976372
    """

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            import subprocess

            to_install = []
            to_install_source_override = []
            for (
                library,
                version,
            ) in libraries.items():  # NOTE: for some reason, list comp breaks this
                if "|" in library:  # signal alternative source url
                    # if specifying the alternative source, always flag to install. this is because the
                    # import_module() step won't fail, resulting in the wrong env
                    library, source = library.split("|")
                    parsed = f"{library}=={version}"
                    to_install_source_override.append((parsed, source))
                    continue
                try:
                    # note: this will fail to import any reqs that have an extra, e.g. rag[serve]. however, we do not
                    # want to ignore the extra or else we will not pip install the extras and we will get downstream
                    # errors
                    import_module(
                        library
                    )  # note: will not throw exception if existing lib is wrong version
                except ModuleNotFoundError:
                    logger.info(f"failed to import library {library}; pip installing")
                    parsed = f"{library}=={version}"
                    to_install.append(parsed)
                except BaseException as e:
                    raise Exception(
                        f"An error occurred while loading module {library}"
                    ) from e

            # without this context manager, you can break your venv if you keyboard interrupt a flow while it's pip
            # installing libraries
            with DelayedKeyboardInterrupt():
                # install directly from pip
                # NOTE: do not use sys.executable, "-m", "pip" because this will pip install to the wrong conda env!
                subprocess.run(["pip", "install", "--quiet", *to_install])
                # install pkgs from remote source
                for pkg in to_install_source_override:
                    parsed, src = pkg
                    print(f"pip installing {parsed} from {src}")
                    subprocess.run(
                        [
                            "pip",
                            "install",
                            # "--ignore-installed",  # force install of remote version
                            "--quiet",
                            parsed,
                            "--find-links",
                            src,
                        ]
                    )

            return function(*args, **kwargs)

        return wrapper

    return decorator


class DelayedKeyboardInterrupt:
    """
    Context manager to prevent keyboardinterrupt from interrupting important code.

    source: https://stackoverflow.com/a/21919644/4212158
    """

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug("SIGINT received. Delaying KeyboardInterrupt.")

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)
