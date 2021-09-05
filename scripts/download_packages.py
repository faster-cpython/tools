# Download N most popular PyPI packages

import json
import os
import argparse

import requests

TOP_PYPI_PACKAGES = (
    "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.json"
)

PYPI_INFO = "https://pypi.python.org/pypi/{}/json"


def dl_data(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.content


def dl_json(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def dl_package_info(package):
    return dl_json(PYPI_INFO.format(package))


parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--number", type=int, default=100, help="How many packages (default 100)"
)
parser.add_argument(
    "-o", "--odir", default="packages", help="Where to download (default ./packages)"
)
parser.add_argument(
    "-t",
    "--top-packages",
    default=TOP_PYPI_PACKAGES,
    help=f"URL for 'top PYPI packages (default {TOP_PYPI_PACKAGES})",
)


def main():
    args = parser.parse_args()
    os.makedirs(args.odir, exist_ok=True)
    packages = dl_json(args.top_packages)
    print("Last update:", packages["last_update"])
    rows = packages["rows"]
    # Sort from high to low download count
    rows.sort(key=lambda row: -row["download_count"])
    # Limit to top N packages
    rows = rows[: args.number]
    print(f"Downloading {len(rows)} packages...")
    index = 0
    count = 0
    skipped = 0
    missing = 0
    try:
        for row in rows:
            print(
                f"Project {row['project']}"
                f" was downloaded {row['download_count']:,d} times"
            )
            index += 1
            info = dl_package_info(row["project"])
            releases = info["releases"]
            # Assume the releases are listed in chronological order
            last_release = list(releases)[-1]
            print(f"  Last release: {last_release}")
            files = releases[last_release]
            for file in files:
                filename = file["filename"]
                # Download the sdist, which is the .tar.gz filename
                if filename.endswith(".tar.gz"):
                    print(f"  File name: {filename}")
                    dest = os.path.basename(filename)
                    fulldest = os.path.join(args.odir, dest)
                    if not os.path.exists(fulldest):
                        url = file["url"]
                        print(f"  URL: {url}")
                        data = dl_data(url)
                        print(f"  Writing {len(data)} bytes to {fulldest} ")
                        with open(fulldest, "wb") as f:
                            f.write(data)
                        count += 1
                    else:
                        print(f"  Skipping {fulldest} (already exists)")
                        skipped += 1
                    break
            else:
                missing += 1
    except KeyboardInterrupt:
        print(f"Interrupted at index {index}")
    finally:
        print(
            f"Out of {len(rows)} packages:"
            f" downloaded {count}, skipped {skipped}, missed {missing}"
        )


if __name__ == "__main__":
    main()
