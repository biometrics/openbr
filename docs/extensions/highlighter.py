import re

from markdown.postprocessors import Postprocessor
from markdown.extensions import Extension

class HighlighterPostprocessor(Postprocessor):
    def run(self, text):
        print text
        return text

class HighlighterExtension(Extension):
    def extendMarkdown(self, md, md_globals):
        md.postprocessors.add('highlight', HighlighterPostprocessor(md))
