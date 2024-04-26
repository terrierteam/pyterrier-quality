# Polyfill for upstream functionality in pyterrier.
# To be removed in a future version.
try:
    from pyterrier.io import finalized_directory, finalized_open, pyterrier_home, download, download_stream, open_or_download_stream, TqdmSha256BufferedReader, TqdmSha256BufferedSequenceReader, Sha256BufferedWriter, path_is_under_base, byte_count_to_human_readable, entry_points
except ImportError:
    import os
    import io
    import tempfile
    import shutil
    import urllib
    import pandas as pd
    from hashlib import sha256
    from contextlib import contextmanager
    from importlib.metadata import EntryPoint
    from importlib.metadata import entry_points as eps
    import pyterrier as pt
    from typing import Optional, Tuple

    DEFAULT_CHUNK_SIZE = 16_384 # 16kb

    @contextmanager
    def finalized_directory(path: str) -> str:
        prefix = f'.{os.path.basename(path)}.tmp.'
        dirname = os.path.dirname(path)
        try:
            path_tmp = tempfile.mkdtemp(prefix=prefix, dir=dirname)
            yield path_tmp
            os.chmod(path_tmp, 0o777) # default directory umask
        except:
            try:
                shutil.rmtree(path_tmp)
            except:
                raise
            raise

        os.replace(path_tmp, path)

    def pyterrier_home() -> str:
        if "PYTERRIER_HOME" in os.environ:
            home = os.environ["PYTERRIER_HOME"]
        else:
            home = os.path.expanduser('~/.pyterrier')
        if not os.path.exists(home):
            os.makedirs(home)
        return home


    def download(url: str, path: str, *, expected_sha256: str = None, verbose: bool = True) -> None:
        with finalized_open(path) as fout, \
             download_stream(url, expected_sha256=expected_sha256, verbose=verbose) as fin:
            while chunk := fin.read1():
                fout.write(chunk)


    @contextmanager
    def download_stream(url: str, *, expected_sha256: Optional[str] = None, verbose: bool = True) -> io.IOBase:
        with urllib.request.urlopen(url) as fin:
            if fin.status == 200:
                total = None
                if 'Content-Length' in fin.headers:
                    total = int(fin.headers['Content-Length'])
                try:
                    tqdm_sha256_in = TqdmSha256BufferedReader(fin, total, desc=url, enabled=verbose)
                    yield tqdm_sha256_in
                finally:
                    tqdm_sha256_in.close()
                if expected_sha256 is not None:
                    if not expected_sha256.lower() == tqdm_sha256_in.sha256.hexdigest().lower():
                        raise ValueError(f'Corrupt download of {url}: expected_sha256={expected_sha256} '
                                         f'but found {tqdm_sha256_in.sha256.hexdigest()}')
            else:
                raise OSError(f'Unhandled status code: {fin.status}')


    @contextmanager
    def open_or_download_stream(
        path_or_url: str,
        *,
        expected_sha256: Optional[str] = None,
        verbose: bool = True
    ) -> io.IOBase:
        if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
            with download_stream(path_or_url, expected_sha256=expected_sha256, verbose=verbose) as fin:
                yield fin
        elif os.path.isfile(path_or_url):
            total = os.path.getsize(path_or_url)
            with open(path_or_url, 'rb') as fin:
                tqdm_sha256_in = TqdmSha256BufferedReader(fin, total, desc=path_or_url, enabled=verbose)
                yield tqdm_sha256_in
        else:
            raise OSError(f'path or url {path_or_url!r} not found')


    class TqdmSha256BufferedReader(io.BufferedIOBase):
        def __init__(self, reader: io.IOBase, total: int, desc: str, enabled: bool = True):
            self.reader = reader
            self.pbar = pt.tqdm(total=total, desc=desc, unit="B", unit_scale=True, unit_divisor=1024, disable=not enabled)
            self.seek = self.reader.seek
            self.tell = self.reader.tell
            self.seekable = self.reader.seekable
            self.readable = self.reader.readable
            self.writable = self.reader.writable
            self.flush = self.reader.flush
            self.isatty = self.reader.isatty
            self.sha256 = sha256()

        def read1(self, size: int = -1) -> bytes:
            if size == -1:
                size = DEFAULT_CHUNK_SIZE
            chunk = self.reader.read1(min(size, DEFAULT_CHUNK_SIZE))
            self.pbar.update(len(chunk))
            self.sha256.update(chunk)
            return chunk

        def read(self, size: int = -1) -> bytes:
            if size == -1:
                size = DEFAULT_CHUNK_SIZE
            chunk = self.reader.read(min(size, DEFAULT_CHUNK_SIZE))
            self.pbar.update(len(chunk))
            self.sha256.update(chunk)
            return chunk

        def close(self) -> None:
            self.pbar.close()
            self.reader.close()


    class TqdmSha256BufferedSequenceReader(io.BufferedIOBase):
        def __init__(self, readers):
            self.readers = readers
            self.x = next(self.readers)
            self.reader = self.x.__enter__()
            self.pbar = self.reader.pbar
            self.seek = self.reader.seek
            self.tell = self.reader.tell
            self.seekable = self.reader.seekable
            self.readable = self.reader.readable
            self.writable = self.reader.writable
            self.flush = self.reader.flush
            self.isatty = self.reader.isatty
            self.close = self.reader.close
            self.sha256 = sha256()

        def read1(self, size: int = -1) -> bytes:
            if size == -1:
                size = DEFAULT_CHUNK_SIZE
            chunk = self.reader.read1(min(size, DEFAULT_CHUNK_SIZE))
            if len(chunk) == 0:
                self.reader.close()
                try:
                    self.x = next(self.readers)
                except StopIteration:
                    return chunk
                self.reader = self.x.__enter__()
                self.pbar = self.reader.pbar
                self.seek = self.reader.seek
                self.tell = self.reader.tell
                self.seekable = self.reader.seekable
                self.readable = self.reader.readable
                self.writable = self.reader.writable
                self.flush = self.reader.flush
                self.isatty = self.reader.isatty
                self.close = self.reader.close
                chunk = self.reader.read1(min(size, DEFAULT_CHUNK_SIZE))
            self.sha256.update(chunk)
            return chunk

        def read(self, size: int = -1) -> bytes:
            chunk = b''
            if size == -1:
                size = DEFAULT_CHUNK_SIZE
            while len(chunk) < size and self.reader is not None:
                chunk += self.reader.read(size - len(chunk))
                if len(chunk) < size:
                    self.reader.close()
                    try:
                        self.x = next(self.readers)
                    except StopIteration:
                        self.x = None
                        self.reader = None
                        return chunk
                    self.reader = self.x.__enter__()
                    self.pbar = self.reader.pbar
                    self.seek = self.reader.seek
                    self.tell = self.reader.tell
                    self.seekable = self.reader.seekable
                    self.readable = self.reader.readable
                    self.writable = self.reader.writable
                    self.flush = self.reader.flush
                    self.isatty = self.reader.isatty
                    self.close = self.reader.close
            self.sha256.update(chunk)
            return chunk


    class Sha256BufferedWriter(io.BufferedIOBase):
        def __init__(self, writer: io.IOBase):
            self.writer = writer
            self.seek = self.writer.seek
            self.tell = self.writer.tell
            self.seekable = self.writer.seekable
            self.readable = self.writer.readable
            self.writable = self.writer.writable
            self.flush = self.writer.flush
            self.isatty = self.writer.isatty
            self.close = self.writer.close
            self.sha256 = sha256()

        def write(self, content: bytes) -> None:
            self.writer.write(content)
            self.sha256.update(content)


    def path_is_under_base(path: str, base: str) -> bool:
        return os.path.realpath(os.path.abspath(os.path.join(base, path))).startswith(os.path.realpath(base))


    def byte_count_to_human_readable(byte_count: float) -> str:
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        while byte_count > 1024 and len(units) > 1:
            byte_count /= 1024
            units = units[1:]
        if units[0] == 'B':
            return f'{byte_count:.0f} {units[0]}'
        return f'{byte_count:.1f} {units[0]}'


    def entry_points(group: str) -> Tuple[EntryPoint, ...]:
        try:
            return tuple(eps(group=group))
        except TypeError:
            return tuple(eps().get(group, tuple()))

    @contextmanager
    def _finalized_open_base(path: str, mode: str, open_fn) -> io.IOBase:
        assert mode in ('b', 't') # must supply either binary or text mode
        prefix = f'.{os.path.basename(path)}.tmp.'
        dirname = os.path.dirname(path)
        try:
            fd, path_tmp = tempfile.mkstemp(prefix=prefix, dir=dirname)
            os.close(fd) # mkstemp returns a low-level file descriptor... Close it and re-open the file the normal way
            with open_fn(path_tmp, f'w{mode}') as fout:
                yield fout
            os.chmod(path_tmp, 0o666) # default file umask
        except:
            try:
                os.remove(path_tmp)
            except:
                raise
            raise

        os.replace(path_tmp, path)


    def finalized_open(path: str, mode: str):
        """
        Opens a file for writing, but reverts it if there was an error in the process.

        Args:
            path(str): Path of file to open
            mode(str): Either t or b, for text or binary mode

        Example:
            Returns a contextmanager that provides a file object, so should be used in a "with" statement. E.g.::

                with pt.io.finalized_open("file.txt", "t") as f:
                    f.write("some text")
                # file.txt exists with contents "some text"

            If there is an error when writing, the file is reverted::

                with pt.io.finalized_open("file.txt", "t") as f:
                    f.write("some other text")
                    raise Exception("an error")
                # file.txt remains unchanged (if existed, contents unchanged; if didn't exist, still doesn't)
        """
        return _finalized_open_base(path, mode, open)
