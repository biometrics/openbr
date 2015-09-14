from ctypes import *
import os

# some notes on ctypes:
# first, you have to make an object that talks directly to the compiled C object
# then, for each function in the C API, the user must define the input/return types
# not setting the argtypes means no args, not setting restypes is void
# there are the normal types, like c_int, c_bool, c_float, c_int_p (char*)
# for char** and *char[], we use POINTER(c_char_p)
# and c_void_p for void*

# these helper functions are just for code reuse -
# they generate common arguments for the  C API functions

def _string_args(n):
    '''
    Returns n char*
    '''
    return [c_char_p]*n

def _var_string_args_func(func, n, *args):
    '''
    Translates the C functions that are
    (int, char**, n char*, and optional extra types in *args)
    to a more pythonic version that wraps the char** arg in
    the appropriate type.
    '''
    def call_func(one, two, *rest):
        arr_type = c_char_p*len(two)
        func.argtypes = [c_int, arr_type] + _string_args(n) + list(args)
        arr = arr_type(*map(c_char_p, two))
        return func(one, arr, *rest)
    return call_func

def _handle_string_func(func, *moretypes):
    '''
    A helper function to make the C functions 
    that populate string buffers more pythonic.
    The functions in the C API return an int 
    (the length of the returned string buffer) 
    and put the return value in a string buffer.
    This function replaces the call to the C obj with two calls - 
        a) one with empty string to get length
        b) one with a string of that length to get the actual value
    Then it returns the string itself. This way, the user of brpy 
    doesn't need to do this stupid magic for themselves. Hooray!
    '''
    func.argtypes = [c_char_p, c_int] + list(moretypes)
    def call_func(*args):
        howlong = func('', 0, *args)
        msg = 'x'*(howlong-1)
        func(msg, howlong, *args)
        return msg
    return call_func

def init_brpy(br_loc='/usr/local/lib'):
    """Initializes all function inputs and outputs for the br ctypes lib object"""

    lib_path = os.environ.get('LD_LIBRARY_PATH')
    paths = [br_loc]
    if lib_path:
        paths.extend(lib_path.split(':'))

    found = False
    for p in paths:
        dylib = '%s/%s.%s' % (p, 'libopenbr', 'dylib')
        so = '%s/%s.%s' % (p, 'libopenbr', 'so')
        if os.path.exists(dylib):
            br = cdll.LoadLibrary(dylib)
            found = True
            break
        elif os.path.exists(so):
            br = cdll.LoadLibrary(so)
            found = True
            break

    if not found:
        raise ValueError('Neither .so nor .dylib libopenbr found in %s' % br_loc)

    br.br_about.restype = c_char_p

    br.br_cat = _var_string_args_func(br.br_cat, 1)

    br.br_cluster = _var_string_args_func(br.br_cluster, 0, c_float, c_char_p)

    br.br_combine_masks = _var_string_args_func(br.br_combine_masks, 2)

    br.br_compare.argtypes = _string_args(3)

    br.br_compare_n = _var_string_args_func(br.br_compare_n, 2)

    br.br_pairwise_compare.argtypes = _string_args(3)

    br.br_convert.argtypes = _string_args(3)

    br.br_enroll.argtypes = _string_args(2)

    br.br_enroll_n = _var_string_args_func(br.br_enroll_n, 1)

    br.br_eval.argtypes = _string_args(3)
    br.br_eval.restype = c_float

    br.br_eval_classification.argtypes = _string_args(4)

    br.br_eval_clustering.argtypes = _string_args(3)

    br.br_eval_detection.argtypes = _string_args(3)
    br.br_eval_detection.restype = c_float

    br.br_eval_landmarking.argtypes = _string_args(3) + [c_int, c_int]
    br.br_eval_landmarking.restype = c_float

    br.br_eval_regression.argtypes = _string_args(4)

    br.br_fuse = _var_string_args_func(br.br_fuse, 3)

    def br_initialize_wrap(func):
        def call_func(argc, argv, sdk_path='', use_gui=False):
            arr_type = c_char_p*len(argv)
            func.argtypes = [POINTER(c_int),arr_type,c_char_p,c_bool]
            arr = arr_type(*map(c_char_p, argv))
            return func(pointer(c_int(argc)), arr, sdk_path, use_gui)
        return call_func
    br.br_initialize = br_initialize_wrap(br.br_initialize)

    br.br_is_classifier.argtypes = [c_char_p]
    br.br_is_classifier.restype = c_bool

    br.br_make_mask.argtypes = _string_args(3)

    br.br_make_pairwise_mask.argtypes = _string_args(3)

    br.br_most_recent_message.restype = c_int
    br.br_most_recent_message = _handle_string_func(br.br_most_recent_message)

    br.br_objects.restype = c_int
    moreargs = _string_args(2) + [c_bool]
    br.br_objects = _handle_string_func(br.br_objects, *moreargs)

    br.br_plot.restype = c_bool
    br.br_plot = _var_string_args_func(br.br_plot, 1, c_bool)

    br.br_plot_detection.restype = c_bool
    br.br_plot_detection = _var_string_args_func(br.br_plot_detection, 1, c_bool)

    br.br_plot_landmarking.restype = c_bool
    br.br_plot_landmarking = _var_string_args_func(br.br_plot_landmarking, 1, c_bool)

    br.br_plot_metadata.restype = c_bool
    br.br_plot_metadata = _var_string_args_func(br.br_plot_metadata, 1, c_bool)

    br.br_progress.restype = c_float

    # TODO: ??? how do *** ???
    br.br_read_pipe.argtypes = [c_char_p, POINTER(c_int), POINTER(POINTER(c_char_p))]
    
    br.br_scratch_path.argtypes = [c_char_p, c_int]
    br.br_scratch_path.restype = c_int
    br.br_scratch_path = _handle_string_func(br.br_scratch_path)

    br.br_sdk_path.restype = c_char_p

    def br_get_header_wrap(func):
        def call_func(matrix, target_gallery, query_gallery):
            arr_type1 = c_char_p*len(target_gallery)
            arr_type2 = c_char_p*len(query_gallery)
            func.argtypes = [c_char_p,arr_type1,arr_type2]
            arr1 = arr_type1(*map(c_char_p, target_gallery))
            arr2 = arr_type2(*map(c_char_p, query_gallery))
            return func(matrix, arr1, arr2)
        return call_func
    br.br_get_header.argtypes = [c_char_p, POINTER(c_char_p), POINTER(c_char_p)]

    br.br_set_header.argtypes = _string_args(3)

    br.br_set_property.argtypes = _string_args(2)

    br.br_time_remaining.restype = c_int

    br.br_train.argtypes = _string_args(2)

    br.br_train_n = _var_string_args_func(br.br_train_n, 1)

    br.br_version.restype = c_char_p

    br.br_slave_process.argtypes = [c_char_p]

    br.br_load_img.argtypes = [c_char_p, c_int]
    br.br_load_img.restype = c_void_p

    br.br_unload_img.argtypes = [c_void_p]
    br.br_unload_img.restype = POINTER(c_ubyte)

    br.br_template_list_from_buffer.argtypes = [c_char_p, c_int]
    br.br_template_list_from_buffer.restype = c_void_p

    br.br_free_template.argtypes = [c_void_p]

    br.br_free_template_list.argtypes = [c_void_p]

    br.br_free_output.argtypes = [c_void_p]

    br.br_img_rows.argtypes = [c_void_p]
    br.br_img_rows.restype = c_int

    br.br_img_cols.argtypes = [c_void_p]
    br.br_img_cols.restype = c_int

    br.br_img_channels.argtypes = [c_void_p]
    br.br_img_channels.restype = c_int

    br.br_img_is_empty.argtypes = [c_void_p]
    br.br_img_is_empty.restype = c_bool

    br.br_get_filename.argtypes = [c_char_p, c_int, c_void_p]
    br.br_get_filename.restype = c_int
    br.br_get_filename = _handle_string_func(br.br_get_filename)

    br.br_set_filename.argtypes = [c_void_p, c_char_p]

    br.br_get_metadata_string.argtypes = [c_char_p, c_int, c_void_p, c_char_p]
    br.br_get_metadata_string.restype = c_int
    br.br_get_metadata_string = _handle_string_func(br.br_get_metadata_string)

    br.br_enroll_template.argtypes = [c_void_p]
    br.br_enroll_template.restype = c_void_p

    br.br_enroll_template_list.argtypes = [c_void_p]
    br.br_enroll_template_list.restype = c_void_p

    br.br_compare_template_lists.argtypes = [c_void_p, c_void_p]
    br.br_compare_template_lists.restype = c_void_p

    br.br_get_matrix_output_at.argtypes = [c_void_p, c_int, c_int]
    br.br_get_matrix_output_at.restype = c_float

    br.br_get_template.argtypes = [c_void_p, c_int]
    br.br_get_template.restype = c_void_p

    br.br_num_templates.argtypes = [c_void_p]
    br.br_num_templates.restype = c_int

    br.br_make_gallery.argtypes = [c_char_p]
    br.br_make_gallery.restype = c_void_p

    br.br_load_from_gallery.argtypes = [c_void_p]
    br.br_load_from_gallery.restype = c_void_p

    br.br_add_template_to_gallery.argtypes = [c_void_p, c_void_p]

    br.br_add_template_list_to_gallery.argtypes = [c_void_p, c_void_p]

    br.br_close_gallery.argtypes = [c_void_p]

    return br
