import os
import markdown
from io import open

from HTMLParser import HTMLParser

def subfiles(path, ext):
    return [os.path.join(path, name) for name in os.listdir(path) if os.path.isfile(os.path.join(path, name)) and name[-len(ext):] == ext]

def subdirs(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def walk(path, ext):
    files = []
    for d in subdirs(path):
        files.extend(walk(os.path.join(path, d), ext))

    files.extend(subfiles(path, ext))
    return files

class Link():
    def __init__(self, path, raw_link):
        if 'http' in raw_link or 'www' in raw_link:
            self.http = raw_link
            self.file = None
            self.anchor = None

        elif raw_link[0] == '#':
            self.http = None
            self.file = None
            self.anchor = raw_link[1:]

        elif '#' in raw_link:
            self.http = None

            split_link = raw_link.split('#')
            self.file = os.path.normpath(os.path.join(path, split_link[0]))
            self.anchor = split_link[1]

        else:
            self.http = None
            self.file = os.path.normpath(os.path.join(path, raw_link))
            self.anchor = None

class LinkParser(HTMLParser):
    def __init__(self, path):
        HTMLParser.__init__(self)

        self.path = path
        self.headers = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        for attr in attrs:
            if u'href' == attr[0]:
                self.links.append(Link(self.path, attr[1].encode('ascii', 'ignore')))

            elif u'class' == attr[0]:
                self.headers.append(attr[1].encode('ascii', 'ignore'))

            elif u'name' == attr[0]:
                self.headers.append(attr[1].encode('ascii', 'ignore'))

            elif u'id' == attr[0]:
                self.headers.append(attr[1].encode('ascii', 'ignore'))


def parse(path, html):
    parser = LinkParser(path)
    parser.feed(html)

    return parser.headers, parser.links

def check(headers, links):
    for f, file_links in links.items():
        for link in file_links:
            if link.http: # Can't check links to other websites
                continue

            link_file = f
            if link.file:
                link_file = link.file

            if link_file.endswith('.pdf'):
                if not os.path.exists(link_file):
                    print 'BAD PDF: ' + link_file + ' DOES NOT EXIST'
                    print
                continue

            if link_file not in headers:
                print 'BAD FILE IN ' + f + ':', link_file
                print
                continue

            if link.anchor and link.anchor != "fnref:1" and link.anchor not in headers[link_file]:
                print 'BAD ANCHOR IN ' + f + ':', link_file + '#' + link.anchor
                print
                continue

def main():
    docs_dir = '../docs/'
    ext = 'md'

    md_files = walk(docs_dir, ext)
    md = markdown.Markdown( ['meta', 'toc', 'tables', 'fenced_code', 'attr_list', 'footnotes'] )

    html_files = [md.convert(open(f, 'r', encoding='utf-8').read()) for f in md_files]

    headers = {}
    links = {}
    for i in range(len(md_files)):
        local_headers, local_links = parse(os.path.dirname(md_files[i]), html_files[i])

        headers[md_files[i]] = local_headers
        links[md_files[i]] = local_links

    check(headers, links)

main()
