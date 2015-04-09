import os
import markdown
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

def basename_no_ext(name):
    basename = os.path.basename(name)
    return basename.split('.')[0]

class Link():
    def __init__(self, raw_link):
        if 'http' in raw_link:
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
            self.file = basename_no_ext(split_link[0])
            self.anchor = split_link[1]
        else:
            self.http = None
            self.file = basename_no_ext(raw_link)
            self.anchor = None

class LinkParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)

        self.headers = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        for attr in attrs:
            if u'href' == attr[0]:
                self.links.append(Link(attr[1].encode('ascii', 'ignore')))

            elif u'class' == attr[0]:
                self.headers.append(attr[1].encode('ascii', 'ignore'))

            elif u'name' == attr[0]:
                self.headers.append(attr[1].encode('ascii', 'ignore'))

            elif u'id' == attr[0]:
                self.headers.append(attr[1].encode('ascii', 'ignore'))


def parse(html):
    parser = LinkParser()
    parser.feed(html)

    return parser.headers, parser.links

def check(headers, links):
    for f, file_links in links.items():
        for link in file_links:
            if link.http:
                continue

            link_file = f
            if link.file:
                link_file = link.file

            if link_file not in headers:
                print 'BAD FILE IN ' + f + '.md:', link_file
                continue

            if link.anchor and link.anchor not in headers[link_file]:
                print 'BAD LINK IN ' + f + '.md:', link_file + ', ' + link.anchor


def main():
    docs_dir = '../docs/'
    ext = 'md'

    md_files = walk(docs_dir, ext)
    md = markdown.Markdown( ['meta', 'toc', 'tables', 'fenced_code'] )

    html_files = [md.convert(open(f, 'r').read()) for f in md_files]

    headers = {}
    links = {}
    for i in range(len(md_files)):
        local_headers, local_links = parse(html_files[i])

        headers[basename_no_ext(md_files[i])] = local_headers
        links[basename_no_ext(md_files[i])] = local_links

    check(headers, links)

main()
