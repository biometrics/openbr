# Python API

The Python API is a light wrapper of the C API. It creates an object that has [all the C functions](c_api/functions.md). Here is a fictional example that, given a `MagicalImageURLGenerator` and a `CatFaceRecognitionModel`, will find cat faces most similar to the cats in a query image.

    from brpy import init_brpy
    import requests # or whatever http request lib you prefer
    import MagicalImageURLGenerator # made up

    # br_loc is /usr/local/lib by default,
    # you may change this by passing a different path to the shared objects
    br = init_brpy(br_loc='/path/to/libopenbr')
    br.br_initialize_default()
    br.br_set_property('algorithm','CatFaceRecognitionModel') # also made up
    br.br_set_property('enrollAll','true')

    mycatsimg = open('mycats.jpg', 'rb').read() # cat picture not provided =^..^=
    mycatstmpl = br.br_load_img(mycatsimg, len(mycatsimg))
    query = br.br_enroll_template(mycatstmpl)
    nqueries = br.br_num_templates(query)

    scores = []
    for imurl in MagicalImageURLGenerator():
        # load and enroll image from URL
        img = requests.get(imurl).content
        tmpl = br.br_load_img(img, len(img))
        targets = br.br_enroll_template(tmpl)
        ntargets = br.br_num_templates(targets)

        # compare and collect scores
        scoresmat = br.br_compare_template_lists(targets, query)
        for r in range(ntargets):
            for c in range(nqueries):
                scores.append((imurl, br.br_get_matrix_output_at(scoresmat, r, c)))

        # clean up - no memory leaks
        br.br_free_template(tmpl)
        br.br_free_template_list(targets)
    
    # print top 10 match URLs
    scores.sort(key=lambda s: s[1])
    for s in scores[:10]:
        print(s[0])

    # clean up - no memory leaks
    br.br_free_template(mycatstmpl)
    br.br_free_template_list(query)
    br.br_finalize()

To enable the module, add `-DBR_INSTALL_BRPY=ON` to your cmake command (or use the ccmake GUI - highly recommended).

Currently only OS X and Linux are supported.
