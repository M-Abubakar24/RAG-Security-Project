"""Download authentic HotpotQA Wikipedia-derived passages and convert to RAG files.
Run: python GET_REAL_RAG_DOCUMENTS.py
Requires only Python standard library. Dataset: Yang et al. (2018), CC BY-SA 4.0.
"""
import csv, json, pathlib, urllib.request

URL = 'https://hotpotqa.github.io/'
DOWNLOADS = [
 'https://raw.githubusercontent.com/hotpotqa/hotpot/master/README.md', # connectivity check not used
]
DATA_URLS = [
 'https://hotpotqa.github.io/', # fallback reference only
 'https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json',
 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json',
]
ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'real_rag_documents'
OUT.mkdir(exist_ok=True)
source = OUT / 'hotpot_dev_distractor_v1.json'
if not source.exists():
    print('Downloading official HotpotQA development dataset (~44 MB). Please wait...')
    try:
        urllib.request.urlretrieve(DATA_URLS[1], source)
    except Exception as e:
        print('Automatic download failed:', e)
        print('Open https://hotpotqa.github.io/ and download Dev set (distractor).')
        print('Place hotpot_dev_distractor_v1.json inside real_rag_documents, then rerun this script.')
        raise SystemExit(1)

with source.open(encoding='utf-8') as f:
    data = json.load(f)

rows = []
seen = set()
# Select 30 real, nonempty source paragraphs across distinct titles, preserving original wording.
for item in data:
    q, answer = item.get('question',''), item.get('answer','')
    for title, sentences in item.get('context',[]):
        content = ''.join(sentences).strip()
        if not content or title in seen or len(content)<120:
            continue
        seen.add(title)
        docid = 'REAL_%03d' % (len(rows)+1)
        rows.append({'doc_id':docid, 'topic':title, 'category':'real_unmodified',
          'attack_style':'none','injection_position':'none','text':content,
          'question':q,'expected_fact':answer,'expected_system_token':'GREEN',
          'attacker_target_token':'','synthetic':'False',
          'source':'HotpotQA dev distractor (Wikipedia-derived)',
          'source_record_id':item.get('_id',''),
          'source_url':'https://hotpotqa.github.io/'})
        (OUT/(docid+'.txt')).write_text(content, encoding='utf-8')
        if len(rows)>=30: break
    if len(rows)>=30: break
if not rows: raise RuntimeError('No source paragraphs were found in the downloaded file.')
fields=list(rows[0].keys())
with (OUT/'real_documents_for_python.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader();writer.writerows(rows)
with (OUT/'SOURCE_AND_LICENSE.txt').open('w',encoding='utf-8') as f:
    f.write('Source: HotpotQA, Yang et al. (2018), EMNLP. https://hotpotqa.github.io/\n')
    f.write('Dataset licence: CC BY-SA 4.0. These are unmodified Wikipedia-derived passages.\n')
    f.write('The question/answer is linked to the source record; it may require multiple passages.\n')
    f.write('Do not treat these documents as attack documents. Create separately labelled copies for injection experiments.\n')
print('SUCCESS: Created',len(rows),'real documents in',OUT)
print('Use real_documents_for_python.csv in your RAG project, or the individual REAL_*.txt files.')
