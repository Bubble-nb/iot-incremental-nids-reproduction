"""Download a fixed public author archive and create disclosed research copies."""
from pathlib import Path
import io, json, hashlib, zipfile, subprocess, sys
import requests
root=Path(__file__).resolve().parent
commit='842737730852c7c7c0955257c48ca72a22770343'
url=f'https://codeload.github.com/jmpr0/incremental-nids/zip/{commit}'
target=root/'original_incremental_nids'
if not target.exists():
    response=requests.get(url,timeout=180)
    response.raise_for_status()
    archive=zipfile.ZipFile(io.BytesIO(response.content))
    prefix=archive.namelist()[0].split('/')[0]+'/'
    files=[]
    for entry in archive.infolist():
        if entry.is_dir(): continue
        relative=Path(entry.filename.removeprefix(prefix))
        destination=(target/relative).resolve()
        if not destination.is_relative_to(target.resolve()):
            raise ValueError('Archive path leaves the target directory')
        payload=archive.read(entry)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(payload)
        files.append(dict(path=relative.as_posix(),sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload)))
    (root/'downloaded_author_manifest.json').write_text(json.dumps(dict(commit=commit,url=url,
        archive_sha256=hashlib.sha256(response.content).hexdigest(),files=files),indent=2),encoding='utf-8')
else:
    print('Author directory already exists; preserving it. Verify its provenance before reuse.')
expected_path=root/'source_manifest.json'
if expected_path.exists():
    expected=json.loads(expected_path.read_text(encoding='utf-8'))
    for item in expected['files']:
        candidate=target/item['path']
        if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('Author file does not match the fixed source manifest: '+item['path'])
    print('Verified all fixed author source hashes.')
if not (root/'compatibility_incremental_nids').exists():
    subprocess.run([sys.executable,str(root/'prepare_smoke.py')],check=True)
if not (root/'research_incremental_nids').exists():
    subprocess.run([sys.executable,str(root/'setup_research.py')],check=True)
print('Prepared author, compatibility and research folders. No training has been started.')
