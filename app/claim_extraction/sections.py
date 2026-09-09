"""Split Markdown at ## headings, leaving each section intact."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    section_id: str
    heading: str
    text: str


@dataclass(frozen=True)
class Report:
    title: str
    sections: list[Section]
    bibliography: str


def split_report(text: str) -> Report:
    headings = []
    fence = ""
    offset = 0
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = ""
        elif marker:
            fence = marker[1]
        else:
            heading = re.match(r"^ {0,3}(#{1,2})[ \t]+(.+?)[ \t]*#*[ \t]*$", line.rstrip("\r\n"))
            if heading:
                headings.append((len(heading[1]), heading[2].strip(), offset, offset + len(line)))
        offset += len(line)

    boundaries = [heading for heading in headings if heading[0] == 2]
    title_heading = next((heading for heading in headings if heading[0] == 1), None)
    title = title_heading[1] if title_heading else "Report"
    if not boundaries:
        return Report(title, [Section("section_0001", title, text)], "")

    sections = []
    bibliography = []
    preamble = text[:boundaries[0][2]]
    body = preamble
    if title_heading and title_heading[2] < boundaries[0][2]:
        body = preamble[:title_heading[2]] + preamble[title_heading[3]:]
    if body.strip():
        sections.append(Section("section_0001", "Preamble", preamble))

    for index, (_, heading, start, _) in enumerate(boundaries):
        end = boundaries[index + 1][2] if index + 1 < len(boundaries) else len(text)
        section_text = text[start:end]
        name = heading.casefold()
        if name in {"references", "bibliography"}:
            bibliography.append(section_text)
        elif name not in {"executive summary", "table of contents"}:
            sections.append(Section(f"section_{len(sections) + 1:04d}", heading, section_text))
    return Report(title, sections, "\n".join(bibliography))
