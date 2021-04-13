# Download N most popular PyPI packages

import json
import os

import requests

TOP_PYPI_PACKAGES = \
    "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.json"

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


def main():
    odir = "packages"  # TODO: Make this an option
    os.makedirs(odir, exist_ok=True)
    packages = dl_json(TOP_PYPI_PACKAGES)
    print("Last update:", packages["last_update"])
    rows = packages["rows"]
    # Sort from high to low download count
    rows.sort(key=lambda row: -row["download_count"])
    # Look at top 100 releases
    for row in rows[:100]:
        print(row)
        info = dl_package_info(row["project"])
        releases = info["releases"]
        # Assume the releases are listed in chronological order
        last_release = list(releases)[-1]
        print(last_release)
        files = releases[last_release]
        for file in files:
            filename = file["filename"]
            # Download the sdist, which is the .tar.gz filename
            if filename.endswith(".tar.gz"):
                print(filename)
                dest = os.path.basename(filename)
                fulldest = os.path.join(odir, dest)
                if not os.path.exists(fulldest):
                    url = file["url"]
                    print(url)
                    data = dl_data(url)
                    print(f"Writing {len(data)} bytes to {fulldest} ")
                    with open(fulldest, "wb") as f:
                        f.write(data)
                else:
                    print(f"Skipping {fulldest} (already exists)")


if __name__ == "__main__":
    main()
