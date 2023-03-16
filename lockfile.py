# coding: UTF-8
import atexit
import datetime
import functools
import os
import random
import re
import tempfile
import time
import weakref
from typing import Any, Callable, List, Optional, TypeVar

R = TypeVar("R") # TODO: use PEP 612 when the framework will be on Python 3.10


class FileLocker:

    INSTANCES_LOCKED = weakref.WeakSet()

    _LOCK_FORMAT = r"{}.{:06d}"
    _LOCK_REGEX_FORMAT = r"{}\.(?P<counter>[0-9]{{6}})"
    _RAND_RANGE = 10e5

    def __init__(self, path: str, timeout: int = 20) -> None:
        self.timeout = timeout

        if os.path.isdir(path):
            self.__folder, self.__filename = path, "locker"

        else:
            self.__folder, self.__filename = os.path.split(os.path.abspath(path))

        self.__pathLocker = None
        self.__id = -1
        self.__isLocked = False
        self.__start = None
        self.__serverTimeLock = None

        if not os.path.isdir(self.__folder):
            raise Exception(f"The specified path is not contained in an existing directory: '{path}'")

    def __enter__(self):
        self.lock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unlock()

    @property
    def _serverTimeLock(self) -> datetime.datetime:
        if self.__serverTimeLock is None:
            tempPath = tempfile.NamedTemporaryFile(dir=self.__folder)
            self.__serverTimeLock = datetime.datetime.fromtimestamp(os.path.getmtime(tempPath.name))
            tempPath.close()

        return self.__serverTimeLock

    def __getLockFileIds(self) -> List[int]:
        pattern = re.compile(FileLocker._LOCK_REGEX_FORMAT.format(self.__filename))
        fileLockerMatches = [fileLockerMatch for fileLockerMatch in map(pattern.match, os.listdir(self.__folder)) if fileLockerMatch]

        # clean-ups lock files which are older than timeout (could be better implemented)
        fullPaths = [os.path.join(self.__folder, fileLockerMatch.string) for fileLockerMatch in fileLockerMatches]
        for path in fullPaths:
            try:
                serverTimeOtherLock = datetime.datetime.fromtimestamp(os.path.getmtime(path))
                if self.timeout < (self._serverTimeLock - serverTimeOtherLock).total_seconds():
                    os.remove(path)

            except OSError:
                # "os.remove": could not occur unless two process try to delete the same file at the same time
                pass

        return [int(fileLockerMatch.group("counter")) for fileLockerMatch in fileLockerMatches]

    def __sleep(self) -> None:
        if self.__start + self.timeout < time.time():
            self.unlock()
            raise TimeoutError(f"Timeout reached after {self.timeout} seconds!")

        time.sleep(0.5)

    def wrapWithDecorator(self, func: Optional[Callable[..., R]] = None, condition: Optional[Callable[..., bool]] = None) -> Callable[..., R]:
        """
        Decorator to wrap any method with a FileLocker.
        It can be used directly without a call or with the 'condition' keyword argument.
        If this method is not used as a decorator, the keyword argument 'func' is necessary.

        Keyword argument 'condition', if defined, must be a function which takes the same arguments/keyword arguments
            than the decorated method and returns True if a lock is necessary.

        Examples of use:

            @fileLocker.wrapWithDecorator
            def myCriticalFunction():
                ...

            @fileLocker.wrapWithDecorator(condition=myCondition):
            def myPotentialyCriticalFunction(arg):
                ...

        Note: the FileLocker instance used is different that the one used to call this method

        """

        def container(method: Callable[..., R]) -> Callable[..., R]:

            @functools.wraps(method)
            def wrapper(*args, **kwargs) -> R:
                if condition and not condition(*args, **kwargs):
                    return method(*args, **kwargs)

                with self.__class__(os.path.join(self.__folder, self.__filename), timeout=self.timeout):
                    return method(*args, **kwargs)

            return wrapper

        if func is None:
            # usecase where the decorator is used by providing the condition such as @wrapWithDecorator(condition=myCondition)
            return container

        # usecase where the decorator is used directly such as @wrapWithDecorator or it is not used as a decorator
        return container(func)

    def wrapWithMonkeyPatch(self, obj: Any, name: str, condition: Optional[Callable[..., bool]] = None) -> None:
        """
        Method to wrap through a monkey-patch the method 'obj.name' with a FileLocker.

        Keyword argument 'condition', if defined, must be a function which takes the same arguments/keyword arguments
            than obj.name and returns True if a lock is necessary.

        By default, a lock is always used.

        """
        method = getattr(obj, name)
        wrapper = self.wrapWithDecorator(func=method, condition=condition)
        setattr(obj, name, wrapper)

    @property
    def locked(self) -> bool:
        return self.__isLocked

    def lock(self) -> None:
        if self.__isLocked:
            raise RuntimeError("Locker already locked")

        FileLocker.INSTANCES_LOCKED.add(self)
        self.__start = time.time()
        self.__isLocked = True
        while self.__isLocked:
            if not self.__getLockFileIds():
                while self.__pathLocker is None or os.path.isfile(self.__pathLocker):
                    self.__id = random.randrange(FileLocker._RAND_RANGE)
                    lockname = FileLocker._LOCK_FORMAT.format(self.__filename, self.__id)
                    self.__pathLocker = os.path.join(self.__folder, lockname)

                try:
                    open(self.__pathLocker, 'w').close()

                except IOError:
                    raise Exception(f"Cannot create the lock file: '{self.__pathLocker}'")

                while True:
                    self.__sleep()
                    if min(self.__getLockFileIds()) == self.__id:
                        self.__isLocked = False
                        break

            else:
                self.__sleep()

    def unlock(self) -> None:
        if self in FileLocker.INSTANCES_LOCKED:
            FileLocker.INSTANCES_LOCKED.remove(self)

        if self.__pathLocker is not None and os.path.isfile(self.__pathLocker):
            os.remove(self.__pathLocker)


def __unlockall() -> None:
    try:
        for fileLocker in FileLocker.INSTANCES_LOCKED:
            if fileLocker and fileLocker.locked:
                fileLocker.unlock()

    except RuntimeError:
        pass # maybe __unlockall()?


atexit.register(__unlockall)
