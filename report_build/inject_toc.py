import re, subprocess, zipfile, shutil, sys

DOCX = "report.docx"; PDF = "report_pages.pdf"; OUT = "report_toc.docx"
z = zipfile.ZipFile(DOCX)
xml = z.read("word/document.xml").decode("utf-8")

# TOC entries: (bookmark id, text)
entries = re.findall(r'<w:hyperlink w:history="1" w:anchor="(_Toc\d+)"><w:r><w:t xml:space="preserve">(.*?)</w:t>', xml)
print(len(entries), "toc entries")

npages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", PDF], capture_output=True, text=True).stdout).group(1))
pages = []
for p in range(1, npages + 1):
    t = subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), "-layout", PDF, "-"], capture_output=True, text=True).stdout
    pages.append([re.sub(r"\s+", "", l) for l in t.splitlines()])

def norm(s):
    return re.sub(r"\s+", " ", s.replace("&amp;", "&")).strip()

ptr = 1  # pdf page index (0-based) to start after TOC
# skip TOC pages: first page whose lines contain a heading line without dot leaders
result = {}
cur = 2
def make_re(text):
    t = norm(text)
    m = re.match(r"บทที่\s*(\d+)", t)
    if m: return re.compile(r"^บท[^\d]{0,8}" + m.group(1) + r"(?![\d.])")
    m = re.match(r"(\d+)\.(\d+)\s", t)
    if m: return re.compile(r"^" + m.group(1) + r"\." + m.group(2) + r"(?!\d)")
    return re.compile(r"^" + re.escape(re.sub(r"\s+", "", t)[:4]))

for bid, text in entries:
    rx = make_re(text)
    found = None
    for pi in range(cur, npages):
        if any(rx.search(l) and ".." not in l for l in pages[pi]):
            found = pi; break
    if found is None:
        print("NOT FOUND:", text); continue
    result[bid] = found
    cur = found
nums = {b: p for b, p in result.items()}  # displayed page = pdf index (0-based) because cover is index 0 -> page 1 at index 1
for bid, text in entries[:6]: print(bid, text, nums.get(bid))
for bid, text in entries[-3:]: print(bid, text, nums.get(bid))

def fix(m):
    bid = m.group(1)
    n = nums.get(bid)
    if n is None: return m.group(0)
    return (f'<w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/><w:instrText xml:space="preserve">PAGEREF {bid} \\h</w:instrText>'
            f'<w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>{n}</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r>')

pat = re.compile(r'<w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/><w:instrText xml:space="preserve">PAGEREF (_Toc\d+) \\h</w:instrText><w:fldChar w:fldCharType="separate"/><w:fldChar w:fldCharType="end"/></w:r>')
new, cnt = pat.subn(fix, xml)
print("patched", cnt)
zo = zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED)
for item in z.infolist():
    data = z.read(item.filename)
    if item.filename == "word/document.xml": data = new.encode("utf-8")
    zo.writestr(item, data)
zo.close()
