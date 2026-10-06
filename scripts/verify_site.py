#!/usr/bin/env python3
"""Check relative resources and anchors without third-party dependencies."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1] / 'docs'

class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links = set(), []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            assert attrs['id'] not in self.ids, f'Duplicate id: {attrs["id"]}'
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])
        if tag == 'img':
            assert attrs.get('alt'), 'Image needs descriptive alt text'

page = Page()
page.feed((ROOT / 'index.html').read_text())
for link in page.links:
    parsed = urlsplit(link)
    if parsed.scheme or parsed.netloc:
        continue
    assert not parsed.path.startswith('/'), f'Use project-relative URL: {link}'
    if parsed.path:
        target = (ROOT / unquote(parsed.path)).resolve()
        assert target.is_relative_to(ROOT), f'Path outside docs: {link}'
        assert target.is_file(), f'Missing file: {link}'
        assert target.stat().st_size > 0, f'Empty file: {link}'
    elif parsed.fragment:
        assert parsed.fragment in page.ids, f'Missing anchor: {link}'
print(f'PASS: {len(page.links)} links/resources; {len(page.ids)} unique anchors.')
