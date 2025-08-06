import re
import json
from pdfminer.high_level import extract_text

def clean_taxonomy_lines(raw_lines):
    """
    1) Skip any blank lines or lines matching header/footer/license patterns.
    2) Build a list of valid_indices = positions of lines that survive step 1.
    3) Iterate over valid_indices in order, keeping track of each line’s raw index.
       a) If a line begins with one or more “....” groups, it’s a new dotted entry:
            – Flush any buffer (if it’s valid taxonomy text), then append “dots + body.”
       b) Otherwise (no leading dots):
            1) If the previous cleaned entry starts with “....” AND the current raw index 
               is exactly previous_raw_index + 1 (i.e. truly adjacent in the original PDF),
               we MERGE this line into that dotted entry (pop/concat/push).
            2) Else (previous wasn’t dotted, or there was a blank/skip in between), peek at 
               the next valid line’s indent‐level:
                 – If next_indent == 1 (exactly “....”), this no‐dot line (plus buffer) is a 
                   candidate root.  But before merging buffer with current, we discard any 
                   buffer that looks like header text.  Then append current (or merged) as root.
                 – Otherwise (next_indent != 1 or no next), treat this line as a broken‐up 
                   root ⇒ append to buffer.
    4) At the end, flush any remaining buffer if it doesn’t look like header text.
    """

    cleaned = []
    buffer = None

    # 1) Patterns of lines to skip entirely (page headers, footers, licenses, etc.)
    skip_pattern = re.compile(
        r"^(?:"
        r"Page\s+\d+"                    # “Page 3”, “Page 10”, etc.
        r"|.*IEEE\s+Taxonomy.*"          # any line containing “IEEE Taxonomy”
        r"|Version\b"                    # “Version …”
        r"|Created\s+by\b"               # “Created by …”
        r"|This\s+work\s+is\s+licensed"  # “This work is licensed …”
        r"|International\s+License"      # “International License …”
        r"|Engineers\s+\(IEEE\)"         # “Engineers (IEEE)” footer
        r"|Creative\s+Commons"           # “Creative Commons …”
        r")"
    )

    # 1b) Any “buffer” that contains these keywords is very likely header/intro text
    header_keywords = re.compile(r"\b(January|IEEE|Thesaurus|branch\))\b", re.IGNORECASE)

    # 2) Build valid_indices = all raw‐line indices whose stripped text is neither blank nor skipped
    valid_indices = []
    for idx, raw in enumerate(raw_lines):
        stripped = raw.strip()
        if not stripped or skip_pattern.match(stripped):
            continue
        valid_indices.append(idx)

    def peek_next_indent(i):
        """
        Return how many “....” groups appear at the very left of raw_lines[j],
        where j is the next valid index after i.  If none, return None.
        """
        nxt = None
        for v in valid_indices:
            if v > i:
                nxt = v
                break
        if nxt is None:
            return None
        m = re.match(r"^\s*(?P<dots>(?:\.\.\.\.)*)", raw_lines[nxt])
        return len(m.group("dots")) // 4

    # 3) Process each valid line in order, tracking the position within valid_indices
    for pos, i in enumerate(valid_indices):
        raw = raw_lines[i]
        stripped = raw.strip()

        # 3a) If this line begins with one or more “....” groups, it’s a new dotted entry.
        m = re.match(r"^\s*(?P<dots>(?:\.\.\.\.)+)\s*(?P<body>.*)$", raw)
        if m:
            # Flush buffer if it’s valid taxonomy text (not header/intro)
            if buffer is not None:
                if not header_keywords.search(buffer):
                    cleaned.append(buffer.strip())
                buffer = None

            dots = m.group("dots")
            body = m.group("body").strip()
            cleaned.append(dots + body)
            continue

        # 3b) No leading dots ⇒ check if the previous cleaned entry was dotted AND truly adjacent
        if cleaned:
            # raw index of previous valid line:
            prev_raw_idx = valid_indices[pos - 1] if pos > 0 else None
            prev_cleaned_line = cleaned[-1]

            # If previous cleaned was a dotted entry (it starts with “....”) and the two lines
            # are adjacent in the original PDF, MERGE:
            if prev_raw_idx is not None \
               and prev_cleaned_line.startswith("....") \
               and (i == prev_raw_idx + 1):
                prev = cleaned.pop()
                merged = prev + " " + stripped
                cleaned.append(merged)
                continue

        # 3c) Otherwise, the previous wasn’t a dotted‐adjacent case.  Peek at next valid indent:
        next_indent = peek_next_indent(i)  # None, 0, 1, 2, …

        if next_indent == 1:
            # Next line is exactly “....something” ⇒ current (plus buffer) is a new root
            if buffer is not None:
                # If buffer looks like header/intro, discard it entirely
                if not header_keywords.search(buffer):
                    merged = buffer + " " + stripped
                    cleaned.append(merged.strip())
                else:
                    cleaned.append(stripped)
                buffer = None
            else:
                cleaned.append(stripped)

        else:
            # Next is either no‐dot (0), deeper indent (≥2), or no next ⇒ broken root continuation
            if buffer is None:
                buffer = stripped
            else:
                buffer += " " + stripped

    # 4) At the very end, flush any remaining buffer if it’s not header text
    if buffer is not None:
        if not header_keywords.search(buffer):
            cleaned.append(buffer.strip())
        buffer = None

    return cleaned


def drop_intro_lines(cleaned):
    """
    Discard any leading lines until we find the first true root category.  A “true root” is:
      • cleaned[i] does NOT start with “....” (i.e. indent_level = 0), AND
      • cleaned[i+1] DOES start with exactly four dots (“....”), AND
      • cleaned[i] does NOT contain obvious header keywords (“January”, “IEEE”, “Thesaurus”).

    Return cleaned[i:] from that point onward; if none matches, return cleaned as-is.
    """
    HEADER_BLACKLIST = ["January", "IEEE", "Thesaurus"]
    for i in range(len(cleaned) - 1):
        curr = cleaned[i]
        nxt  = cleaned[i + 1]
        if (not curr.startswith("....")
            and nxt.startswith("....")
            and not any(keyword.lower() in curr.lower() for keyword in HEADER_BLACKLIST)):
            return cleaned[i:]
    return cleaned


def parse_taxonomy(lines):
    """
    Build a nested JSON from the cleaned lines.  Each line’s indent_level = (# of leading “....” groups).
    Strip off those dots to get “name,” then attach each node under its parent (indent_level – 1).
    """
    root = []
    stack = []

    for line in lines:
        m = re.match(r"^(?P<dots>(?:\.\.\.\.)*)", line)
        indent_level = len(m.group("dots")) // 4
        term = line.lstrip(".").strip()

        node = {"name": term, "children": []}
        if indent_level == 0:
            root.append(node)
            stack = [node]
        else:
            while len(stack) > indent_level:
                stack.pop()
            stack[-1]["children"].append(node)
            stack.append(node)

    return root


if __name__ == "__main__":
    pdf_path = "ieee-taxonomy.pdf"   # ← adjust path if needed
    text = extract_text(pdf_path)
    raw_lines = text.splitlines()

    cleaned = clean_taxonomy_lines(raw_lines)
    taxonomy_only = drop_intro_lines(cleaned)
    taxonomy_tree = parse_taxonomy(taxonomy_only)

    with open("ieee_taxonomy_clean.json", "w", encoding="utf-8") as f:
        json.dump(taxonomy_tree, f, indent=4, ensure_ascii=False)

    print("Taxonomy JSON created successfully!")

















