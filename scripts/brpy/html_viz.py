'''
Some funcs to generate HTML visualizations.
Run from the folder you intend to save the HTML page so
the relative paths in the HTML file are correct and PIL can find the images on disk.
Requires local images, but should be pretty easy to set up an apache server (or whatev)
and host them as long as the relative paths remain the same on ya serva.
'''

from PIL import Image

def crop_to_bb(x, y, width, height, imname, maxheight=None):
    '''
    Generates an HTML string that crops to a given bounding box and resizes to maxheight pixels.
    A maxheight of None will keep the original size (default).
    When two crops are put next to each other, they will be inline. To make each crop its own line, wrap it in a div.
    '''
    img = Image.open(imname)
    imwidth, imheight = img.size
    if not maxheight:
        maxheight = height
    ratio = maxheight / height
    # note for future me:
    # image is cropped with div width/height + overflow:hidden,
    # resized with img height,
    # and positioned with img margin
    html = '<div style="overflow:hidden; display:inline-block; width:%ipx; height:%ipx;">' % (width*ratio, maxheight)
    html += '<img src="%s" style="height:%ipx; margin:-%ipx 0 0 -%ipx;"/>' % (imname, imheight*ratio, y*ratio, x*ratio)
    html += '</div>'
    return html

def bbs_for_image(imname, bbs, maxheight=None, colors=None):
    '''
    Generates an HTML string for an image with bounding boxes.
    bbs: iterable of (x,y,width,height) bounding box tuples
    '''
    img = Image.open(imname)
    imwidth, imheight = img.size
    if not maxheight:
        maxheight = imheight
    ratio = maxheight/imheight
    html = [
            '<div style="position:relative">',
            '<img src="%s" style="height:%ipx" />' % (imname, maxheight)
           ]
    if not colors:
        colors = ['green']*len(bbs)
    html.extend([ bb(*box, ratio=ratio, color=color) for color,box in zip(colors,bbs) ])
    html.append('</div>')
    return '\n'.join(html)


def bb(x, y, width, height, ratio=1.0, color='green'):
    '''
    Generates an HTML string bounding box.
    '''
    html = '<div style="position:absolute; border:2px solid %s; color:%s; left:%ipx; top:%ipx; width:%ipx; height:%ipx;"></div>'
    return html % (color, color, x*ratio, y*ratio, width*ratio, height*ratio)
