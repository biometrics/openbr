import os
import re

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

    blocks = docsMatch.group().split('\\')[1:]
    if len(blocks) == 0:
        return None

    attributes = {}
    for block in blocks:
        key = block[:block.find(' ')]
        value = block[block.find(' '):].split('\n')[0].strip()
        if key in attributes:
            attributes[key].append(value)
        else:
            attributes[key] = [value]

    attributes['Name'] = clss[5:clss.find(':')].strip()
    attributes['Parent'] = clss[clss.find('public')+6:].strip()
    return attributes

def parseSees(sees):
    if not sees:
        return ""

    output = "* **see:**"
    if len(sees) > 1:
        output += "\n\n"
        for see in sees:
            output += "\t* [" + see + "](" + see + ")\n"
        output += "\n"
    else:
        output += " [" + sees[0] + "](" + sees[0] + ")\n"

    return output

def parseAuthors(authors):
    if not authors:
        return "* **authors:** None\n"

    output = "* **author"
    if len(authors) > 1:
        output += "s:** " + ", ".join(authors) + "\n"
    else:
        output += ":** " + authors[0] + "\n"

    return output

def parseProperties(properties):
    if not properties:
        return "* **properties:** None\n\n"

    output = "* **properties:**\n\n"
    output += "Name | Type | Description\n"
    output += "--- | --- | ---\n"
    for prop in properties:
        split = prop.split(' ')
        ty = split[0]
        name = split[1]
        desc = ' '.join(split[2:])

        output += name + " | " + ty + " | " + desc + "\n"

    return output

def main():
    plugins_dir = '../../openbr/plugins/'
    output_dir = '../docs/docs/plugins/'

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

                plugin_string = "# " + attributes["Name"] + "\n\n"
                plugin_string += ' '.join([brief for brief in attributes["brief"]]) + "\n\n"
                plugin_string += "* **file:** " + os.path.join(module, plugin) + "\n"
                plugin_string += "* **inherits:** [" + attributes["Parent"] + "](../cpp_api.md#" + attributes["Parent"].lower() + ")\n"

                plugin_string += parseSees(attributes.get("see", None))
                plugin_string += parseAuthors(attributes.get("author", None))
                plugin_string += parseProperties(attributes.get("property", None))

                plugin_string += "\n---\n\n"

                names.append(attributes["Name"])
                docs[attributes["Name"]] = plugin_string

        for name in sorted(names):
            output_file.write(docs[name])
main()
