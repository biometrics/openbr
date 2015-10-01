#!/usr/bin/env python

import csv, sys, json, argparse
from brpy.html_viz import crop_to_bb

parser = argparse.ArgumentParser(description='Visualize face cluster results in an HTML page.')
parser.add_argument('input_file', type=str, help='Results from clustering (in csv format)')
parser.add_argument('img_loc', type=str, help='Location of images on disk (this gets prepended to image path)')
parser.add_argument('--cluster_key', '-c', type=str, default='Cluster', help='The name of the cluster ID header in input_file')
parser.add_argument('--height', '-mh', type=int, default=400, help='Height of the cluster rows in pixels')
parser.add_argument('--output_file', '-o', type=str, default='clustahs.html', help='Where to save the output HTML file.')
args = parser.parse_args()

maxheight = args.height
clustmap = dict()
with open(args.input_file) as f:
    for line in csv.DictReader(f):
        c = int(line[args.cluster_key])
        if c not in clustmap:
            clustmap[c] = []
        x,y,width,height = [ float(line[k]) for k in ('Face_X','Face_Y','Face_Width','Face_Height') ]
        imname = '%s/%s' % (args.img_loc, line['File'])
        try:
            html = crop_to_bb(x,y,width,height,imname,maxheight=400)
        except IOError:
            print('problem with %s' % imname)
            continue
        clustmap[c].append(html)

# browsers crash for a DOM with this many img tags,
# so instead this makes links for each cluster that dynamically populate the DOM using the HTML strings from
# the clusters javascript object below (which is a direct translation of clustmap above).
# hacky, but it works.
html = ['<!DOCTYPE html>', '<html>', '<head>', '<title>Face clusters</title>', '</head>', '<body>']
script = '''
<script type="text/javascript">
var clusters = %s;
function showCluster(c) {
    var el = document.createElement('div');
    el.innerHTML = clusters[c].join('');
    document.getElementById('cluster'+c).appendChild(el);
}
</script>
'''
html.append(script % json.dumps(clustmap))
for c, imgs in clustmap.items():
    html.append('<div id="cluster%i" style="white-space:nowrap;">' % c)
    reveal = '<a href="javascript:showCluster(\'%i\');">show cluster %i (count=%i)</a>'
    html.append(reveal % (c,c,len(imgs)))
    html.append('</div>')
html.extend(['</body>','</html>'])

with open(args.output_file, 'w') as out:
    out.write("\n".join(html))
