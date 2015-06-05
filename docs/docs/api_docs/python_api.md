# Python API

The Python API is a light wrapper of the C API. It creates an object that has [all the C functions](c_api/functions.md):

    from brpy import init_brpy

    # br_loc is /usr/local/lib by default,
    # you may change this by passing a different path to the shared objects
    br = init_brpy(br_loc='/path/to/libopenbr')
    br.br_initialize_default()

    img = open('catpic.jpg','rb').read()

    br.br_set_property('algorithm','MyCatFaceDetectionModel')
    br.br_set_property('enrollAll','true')

    tmpl = br.br_load_img(img, len(img))
    catfaces = br.br_enroll_template(tmpl)

    print('This pic has %i cats in it!' % br.br_num_templates(catfaces))

    br.br_free_template(tmpl)
    br.br_free_template_list(catfaces)
    br.br_finalize()

To enable the module, add `-DBR_INSTALL_BRPY=ON` to your cmake command (or use the ccmake GUI - highly recommended).

Currently only OS X and Linux are supported.
