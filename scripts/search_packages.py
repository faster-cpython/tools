import argparse
import contextlib
import glob
import re
import sys
import tarfile


class Reporter:
    def __init__(self, extensions: list[str], query: str, verbose: int):
        self.extensions = tuple(extensions)
        # query is a bytes regex since tarfile gives us bytes
        self.query = re.compile(query.encode("UTF-8"))
        self.verbose = verbose

    def tarball_report(self, filename: str):
        hits = 0
        hit_files: set[str] = set()
        if self.verbose > 1:
            print(f"\nExamining tarball {filename}")
        try:
            tar = tarfile.open(filename, "r")
        except OSError:
            if self.verbose > 0:
                print("{filename}: Cannot open tarball")
            return
        with contextlib.closing(tar):
            members = tar.getmembers()
            for m in members:
                info = m.get_info()
                name = info["name"] if isinstance(info["name"], str) else ""
                ext = name.rpartition(".")[2]
                if ext in self.extensions:
                    try:
                        f = tar.extractfile(m)
                        if f is None:
                            if self.verbose > 0:
                                print(f"{name}: Cannot extract")
                            continue
                        source = f.read()
                    except Exception as err:
                        if self.verbose > 0:
                            print(f"{name}: {err}")
                        continue
                    if self.verbose > 2:
                        print(f"Searching {name}")
                    lines = source.splitlines()
                    for i, line in enumerate(lines, start=1):
                        if (m := self.query.search(line)) is not None:
                            if self.verbose > 0:
                                print(
                                    f"{name}:{i}: {line.decode('UTF-8', errors='replace')}"
                                )
                            hits += 1
                            hit_files.add(name)
        if hits and self.verbose >= 0:
            print(f"{filename}: {hits} hits in {len(hit_files)} files")
            if self.verbose == 1:
                print()


def expand_globs(filenames: list[str]) -> str:
    for filename in filenames:
        if "*" in filename and sys.platform == "win32":
            for fn in glob.glob(filename):
                yield fn
        else:
            yield filename


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "-q", "--quiet", action="count", default=0, help="Print less output"
)
argparser.add_argument(
    "-v", "--verbose", action="count", default=0, help="Print more output"
)
argparser.add_argument(
    "-x",
    "--extensions",
    default="py",
    help="Comma-separated list of extensions (default: 'py')",
)
argparser.add_argument("regex", help="Search regular expression")
argparser.add_argument(
    "filenames",
    nargs="*",
    metavar="FILE",
    help="files, directories or tarballs to count",
)


def main():
    args = argparser.parse_args()
    verbose = 1 + args.verbose - args.quiet
    r = Reporter(args.extensions.split(","), args.regex, verbose)
    for file in expand_globs(args.filenames):
        assert (
            file.endswith(".tar") or file.endswith(".tgz") or file.endswith(".tar.gz")
        ), file
        r.tarball_report(file)


if __name__ == "__main__":
    main()
