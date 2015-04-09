# This script is intended to take an API-style class from OpenBR and output
# structured Markdown text in the OpenBR documentation format. The class details
# should then be filled in.

import sys
import re

def find_name_parent(content):
    name_re = re.compile('class(.*?):')
    name = name_re.search(content).group()[:-1].strip().split(' ')[-1]

    parent_re = re.compile('public.*')
    parent = parent_re.search(content).group().strip().split(' ')[-1]

    return name, parent

def find_properties(content):
    properties = []

    property_re = re.compile('Q_PROPERTY\((.*?)\)')
    it = property_re.finditer(content)
    for prop in it:
        prop_type = prop.group()[11:-1].split(' ')[0]
        prop_name = prop.group()[11:-1].split(' ')[1]
        properties.append((prop_name, prop_type))

    return properties

def find_members(content):
    members = []

    member_re = re.compile('(\w+(<\w+>)*)\s(\w+);')
    it = member_re.finditer(content)
    for member in it:
        member_type = member.group()[:-1].strip().split(' ')[0]
        member_name = member.group()[:-1].strip().split(' ')[1]
        members.append((member_name, member_type))

    return members

def find_constructors(name, content):
    constructors = []

    constructor_re = re.compile('(.*)' + name + '\(\)')
    it = constructor_re.finditer(content)
    for constructor in it:
        constructors.append(constructor.group().strip())

    return constructors

def find_functions(content):
    functions = []

    function_re = re.compile('(.*)[^PROPERTY]\(.*\).*;')
    it = function_re.finditer(content)
    for func in it:
        function = {}

        function['Full'] = func.group()[:-1].strip() #don't include semi colon

        func_split = func.group().strip().split(' ')
        if func_split[0] == "static":
            function['Type'] = 'static'
            function['Return'] = func_split[1]
            function['Name'] = func_split[2].split('(')[0]
        elif func_split[0] == "virtual":
            function['Type'] = 'virtual'
            function['Return'] = func_split[1]
            function['Name'] = func_split[2].split('(')[0]
        else:
            function['Type'] = 'normal'
            function['Return'] = func_split[0]
            function['Name'] = func_split[1].split('(')[0]

        args = []

        args_list = func.group()[func.group().find('(')+1:func.group().find(')')].split(',')
        for arg in args_list:
            arg = arg.strip()
            split_idx = arg.rfind(' ')
            if arg[split_idx:].strip()[0] == '*' or arg[split_idx:].strip()[0] == '&':
                split_idx += 2

            args.append((arg[split_idx:].strip(), arg[:split_idx].strip()))

        function['Args'] = args

        functions.append(function)

    return functions

def parse(content):
    name, parent = find_name_parent(content)
    properties = find_properties(content)
    members = find_members(content)
    constructors = find_constructors(name, content)
    functions = find_functions(content)

    return name, parent, properties, members, constructors, functions

def function_builder(name, function):
    markdown = ""

    markdown += "### <h3 id=" + name.lower() + "-function-" + function['Name'].lower() + ">" + function['Full'] + "</h3>\n\n"
    markdown += "DOCUMENT ME\n\n"

    markdown += "* **function definition:**\n\n"
    markdown += "\t\t" + function['Full'] + "\n\n"

    markdown += "* **parameters:**"
    if len(function['Args']) == 0:
        markdown += " NONE\n"
    else:
        markdown += "\n\n"
        markdown += "\tParameter | Type | Description\n"
        markdown += "\t--- | --- | ---\n"
        for arg in function['Args']:
            markdown += "\t" + arg[0] + " | " + arg[1] + " | DOCUMENT ME\n"
        markdown += "\n"

    markdown += "* **output:** (" + function['Return'] + ") DOCUMENT ME\n\n\n"

    return markdown

def format_md(name, parent, properties, members, constructors, functions):
    markdown = ""

    markdown += "# " + name + "\n\n"
    markdown += "Inherits from [" + parent + "](#" + parent.lower() + ")\n\n"

    markdown += "## <h2 id=" + name.lower() + "-properties>Properties</h2>\n\n"
    if len(properties) == 0:
        markdown += "NONE\n\n"
    else:
        markdown += "Property | Type | Description\n"
        markdown += "--- | --- | ---\n"
        for prop in properties:
            markdown += '<a class="table-anchor" id=' + name.lower() + '-properties-' + prop[0].lower() + '></a>'
            markdown += prop[0] + " | " + prop[1] + " | DOCUMENT ME\n"
        markdown += "\n"

    markdown += "## <h2 id=" + name.lower() + "-members>Members</h2>\n\n"
    if len(members) == 0:
        markdown += "NONE\n\n"
    else:
        markdown += "Member | Type | Description\n"
        markdown += "--- | --- | ---\n"
        for member in members:
            markdown += '<a class="table-anchor" id=' + name.lower() + '-members-' + member[0].lower() + '></a>'
            markdown += member[0] + " | " + member[1] + " | DOCUMENT ME\n"
        markdown += "\n"

    markdown += "## <h2 id=" + name.lower() + "-constructors>Constructors</h2>\n\n"
    if len(constructors) == 0:
        markdown += "NONE\n\n"
    else:
        markdown += "Constructor \| Destructor | Description\n"
        markdown += "--- | ---\n"
        for constructor in constructors:
            markdown += constructor + " | DOCUMENT ME\n"
        markdown += "\n"

    markdown += "## <h2 id=" + name.lower() + "-static-functions>Static Functions</h2>\n\n"
    for function in functions:
        if function['Type'] == 'static':
            markdown += function_builder(name, function)

    markdown += "## <h2 id=" + name.lower() + "-functions>Functions</h2>\n\n"
    for function in functions:
        if not function['Type'] == 'static':
            markdown += function_builder(name, function)

    return markdown

def main():
    if len(sys.argv) != 3:
        print 'Inputs => class documentation'
        sys.exit(1)

    class_file = open(sys.argv[1], 'r')
    doc_file = open(sys.argv[2], 'w+')

    name, parent, properties, members, constructors, functions = parse(class_file.read())

    doc_file.write(format_md(name, parent, properties, members, constructors, functions))

main()
