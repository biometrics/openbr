import os, sys
import re

abstractions = ['FileList', 'File',
                'TemplateList', 'Template',
                'UntrainableTransform',
                'MetaTransform', 'UntrainableMetaTransform',
                'MetadataTransform', 'UntrainableMetadataTransform',
                'TimeVaryingTransform',
                'CompositeTransform', 'WrapperTransform',
                'Transform',
                'UntrainableDistance', 'Distance',
                'MatrixOutput', 'Output',
                'Format',
                'FileGallery', 'Gallery',
                'Representation',
                'Classifier'
               ]

def subfiles(path):
    return [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name)) and not name[0] == '.']

def subdirs(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def formatModule(module):
    if module == 'io':
        return 'i/o'
    else:
        return module.capitalize()

def parse(group):
    docs = re.compile('/\*\!(.*?)\*/', re.DOTALL)

    docsMatch = docs.match(group)
    clss = group[docsMatch.end():].strip()
    if len(clss) == 0 or 'class' not in clss:
        return None

    clean_docs = docsMatch.group()[3:-2].replace('*', ' ').strip()
    blocks = clean_docs.split('\\')[1:]
    if len(blocks) == 0:
        return None

    attributes = {}
    for block in blocks:
        key = block[:block.find(' ')]
        value = block[block.find(' '):].strip()
        if key in attributes:
            attributes[key].append(value)
        else:
            attributes[key] = [value]

    attributes['Name'] = clss[5:clss.find(':')].strip()
    attributes['Parent'] = clss[clss.find('public')+6:].strip().strip(',') # Handles the edge case of multiple inheritence
    return attributes

def parseBrief(briefs):
    if not briefs:
        print 'All plugins need a description!'
        sys.exit(1)

    output = ""
    for brief in briefs:
        for abstraction in abstractions:
            regex = re.compile('\s' + abstraction + "'?s?")

            matches = regex.finditer(brief)
            for match in matches:
                name = ' [' + abstraction + ']'
                link = '(../cpp_api/' + abstraction.lower() + '/' + abstraction.lower() + '.md)'
                brief = brief.replace(match.group(), name + link).strip() # strip removes a possible leading whitespace

        for line in brief.split('\n'):
            if not line.strip():
                output += "\n"
            else:
                output += line.strip() + "\n"

        output += ' '

    return output + '\n\n'

def parseInheritance(inheritance):
    if inheritance in abstractions:
        return '../cpp_api/' + inheritance.lower() + '/' + inheritance.lower() + '.md'
    else: # Not an abstraction must inherit in the local file!
        return '#' + inheritance.lower()

def parseLinks(links):
    if not links:
        return ""

    output = "* **see:**"
    if len(links) > 1:
        output += "\n\n"
        for link in links:
            output += "\t* [" + link + "](" + link + ")\n"
        output += "\n"
    else:
        output += " [" + links[0] + "](" + links[0] + ")\n"

    return output

def parsePapers(papers):
    if not papers:
        return ""

    output = "* **read:**\n\n"
    for i in range(len(papers)):
        info = papers[i].split('\n')
        authors = info[0].strip()
        title = None
        other = None
        if len(info) >= 2:
            title = info[1].strip()
        if len(info) >= 3:
            other = info[2].strip()

        output += "\t" + str(i+1) + ". *" + authors + "*\n"
        if title:
            output += "\t **" + title + "**\n"
        if other:
            output += "\t " + other + "\n"
        output += "\n"

    return output

def parseAuthors(authors, citations):
    if not authors:
        print 'All plugins need an author!'
        sys.exit(1)

    if len(authors) != len(citations):
        print 'Different number of authors and citations!'
        print authors, citations
        return "* **authors:** PARSING ERROR\n"

    output = "* **author(s):** "
    for i in range(len(authors)):
        output += "[" + authors[i] + "][" + citations[i] + "], "
    output = output[:-2] + "\n"

    return output

def parseProperties(properties):
    if not properties:
        return "* **properties:** None\n\n"

    output = "* **properties:**\n\n"
    output += "\tProperty | Type | Description\n"
    output += "\t--- | --- | ---\n"
    for prop in properties:
        split = prop.split(' ')
        ty = split[0]
        name = split[1]
        desc = ' '.join(split[2:])

        table_regex = re.compile('\[(.*?)\]')
        table_match = table_regex.search(desc)
        while table_match:
            before = desc[:table_match.start()]
            after = desc[table_match.end():]

            table_content = desc[table_match.start()+1:table_match.end()-1].split(',')

            table = "<ul>"
            for field in table_content:
                table += "<li>" + field.strip() + "</li>"
            table += "</ul>"

            desc = before.strip() + table + after.strip()
            table_match = table_regex.search(desc)

        output += "\t" + name + " | " + ty + " | " + desc + "\n"

    return output

def parseFormats(formats):
    if not formats:
        return ""

    output = "* **format:** "
    for f in formats:
        in_raw = False
        for line in f.split('\n'):
            if not line.strip():
                output += "<pre><code>" if not in_raw else "</code></pre>"
                in_raw = not in_raw
                continue

            clean_line = line.strip()

            # <> are often used describing formats. Unfortunately they are also html tags.
            # So we need to replace them with safe alternatives
            if '<' in clean_line:
                clean_line = clean_line.replace('<', '&lt;')
            if '>' in clean_line:
                clean_line = clean_line.replace('>', '&gt;')

            output += clean_line
            if in_raw:
                output += "\n"
            else:
                output += ' '

        if in_raw:
            output += "</code></pre>"

    return output + "\n"

def main():
    plugins_dir = '../../openbr/plugins/'
    output_dir = '../docs/api_docs/plugins/'

    for module in subdirs(plugins_dir):
        if module == "cmake":
            continue

        output_file = open(os.path.join(output_dir, module + '.md'), 'w+')

        names = []
        docs = {} # Store the strings here first so they can be alphabetized

        for plugin in subfiles(os.path.join(plugins_dir, module)):
            f = open(os.path.join(os.path.join(plugins_dir, module), plugin), 'r')
            content = f.read()

            regex = re.compile('/\*\!(.*?)\*/\n(.*?)\n', re.DOTALL)
            it = regex.finditer(content)
            for match in it:
                attributes = parse(match.group())
                if not attributes or (attributes and attributes["Parent"] == "Initializer"):
                    continue

                print 'documenting ' + module + "/" + plugin + '...'

                plugin_string = "# " + attributes["Name"] + "\n\n"
                plugin_string += parseBrief(attributes.get("brief", None))
                plugin_string += "* **file:** " + os.path.join(module, plugin) + "\n"
                plugin_string += "* **inherits:** [" + attributes["Parent"] + "](" + parseInheritance(attributes["Parent"]) + ")\n"

                plugin_string += parseAuthors(attributes.get("author", None), attributes.get("cite", None))
                plugin_string += parseLinks(attributes.get("br_link", None))
                plugin_string += parsePapers(attributes.get("br_paper", None))
                plugin_string += parseFormats(attributes.get("br_format", None))
                plugin_string += parseProperties(attributes.get("br_property", None))

                plugin_string += "\n---\n\n"

                names.append(attributes["Name"])
                docs[attributes["Name"]] = plugin_string

        for name in sorted(names):
            output_file.write(docs[name])
main()
