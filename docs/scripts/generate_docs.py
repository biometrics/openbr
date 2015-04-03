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

def parseProperties(properties):
    output = ""
    for prop in properties:
        split = prop.find(' ')
        name = prop[:split]
        desc = prop[split:]

        output += "\t* **" + name + "**- " + desc + "\n"
    return output

def main():
    plugins_dir = '../../openbr/plugins/'
    output_dir = '../docs/docs/plugins/'

    for module in subdirs(plugins_dir):
        if module == "cmake":
            continue

        output_file = open(os.path.join(output_dir, module + '.md'), 'w+')

        for plugin in subfiles(os.path.join(plugins_dir, module)):
            f = open(os.path.join(os.path.join(plugins_dir, module), plugin), 'r')
            content = f.read()

            regex = re.compile('/\*\!(.*?)\*/\n(.*?)\n', re.DOTALL)
            it = regex.finditer(content)
            for match in it:
                attributes = parse(match.group())
                if not attributes:
                    continue

                output_file.write("---\n\n")
                output_file.write("# " + attributes["Name"] + "\n\n")
                output_file.write(attributes["brief"][0] + "\n\n")
                output_file.write("* **file:** " + os.path.join(module, plugin) + "\n")
                output_file.write("* **inherits:** [" + attributes["Parent"] + "](../cpp_api.md#" + attributes["Parent"].lower() + ")\n")

                authors = attributes["author"]
                if len(authors) > 1:
                    output_file.write("* **authors:** " + ", ".join(attributes["author"]) + "\n")
                else:
                    output_file.write("* **author:** " + attributes["author"][0] + "\n")

                if not 'property' in attributes.keys():
                    output_file.write("* **properties:** None\n")
                else:
                    properties = attributes['property']
                    output_file.write("* **properties:**\n\n")
                    output_file.write(parseProperties(properties) + "\n")

                output_file.write("\n")
main()
