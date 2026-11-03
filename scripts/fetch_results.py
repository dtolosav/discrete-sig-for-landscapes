"""Download and verify the versioned reporting bundle; no experiments are run."""
import hashlib
import json
from pathlib import Path
import tarfile
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]


def main():
    release = json.loads((ROOT / 'configs/release.json').read_text())
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    archive = output / release['filename']
    if not archive.exists():
        urlretrieve(release['url'], archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == release['sha256'], 'Result bundle checksum mismatch'
    with tarfile.open(archive) as tar:
        tar.extractall(output, filter='data')
    print(output / release['directory'])


if __name__ == '__main__':
    main()
