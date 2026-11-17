REAL DOCUMENTS FOR YOUR RAG PROJECT

1. Download GET_REAL_RAG_DOCUMENTS.py.
2. Put it in your RAG-Security-Project folder.
3. Open a terminal in that folder.
4. Run: python GET_REAL_RAG_DOCUMENTS.py
   (On Windows, if python fails, try: py GET_REAL_RAG_DOCUMENTS.py)
5. A folder named real_rag_documents will appear.
6. Inside are 30 real .txt documents plus real_documents_for_python.csv.
7. Point your existing RAG program to that CSV or to the text files.

This script fetches the official HotpotQA development dataset, and selects
30 unmodified Wikipedia-derived paragraphs from distinct article titles.
Source: https://hotpotqa.github.io/ (Yang et al., EMNLP 2018)
License: CC BY-SA 4.0. Attribution is required.

The downloaded real documents do not contain malicious instructions.
For GREEN/BLUE experiments, make separate labelled modified copies and
compare results against these untouched originals.

If the official download fails, use the 'Dev set (distractor)' link on the
HotpotQA site and place the downloaded JSON in real_rag_documents before rerunning.
