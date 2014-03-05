from ctypes import *
import os

def _string_args(n):
	s  = []
	for i in range(n):
		s.append(c_char_p)
	return s

def _var_string_args(n):
	s = [c_int, POINTER(c_char_p)]
	s.extend(_string_args(n))
	return s

def _handle_string_func(func):
    def call_func(*args):
        howlong = func('', 0, *args)
        msg = 'x'*(howlong-1)
        func(msg, howlong, *args)
        return msg
    return call_func

def init_brpy(br_loc='/usr/local/lib'):
    """Takes the ctypes lib object for br and initializes all function inputs and outputs"""
    br_loc += '/libopenbr.%s'
    if os.path.exists(br_loc % 'dylib'):
        br = cdll.LoadLibrary(br_loc % 'dylib')
    elif os.path.exists(br_loc % 'so'):
        br = cdll.LoadLibrary(br_loc % 'so')
    else:
        raise ValueError('Neither .so nor .dylib libopenbr found in %s' % br_loc)

    plot_args = _var_string_args(1) + [c_bool]
    br.br_about.restype = c_char_p
    br.br_cat.argtypes = _var_string_args(1)
    br.br_cluster.argtypes = [c_int, POINTER(c_char_p), c_float, c_char_p]
    br.br_combine_masks.argtypes = _var_string_args(2)
    br.br_compare.argtypes = _string_args(3)
    br.br_compare_n.argtypes = [c_int, POINTER(c_char_p)] + _string_args(2)
    br.br_pairwise_compare.argtypes = _string_args(3)
    br.br_convert.argtypes = _string_args(3)
    br.br_enroll.argtypes = _string_args(2)
    br.br_enroll_n.argtypes = _var_string_args(1)
    br.br_eval.argtypes = _string_args(3)
    br.br_eval.restype = c_float
    br.br_eval_classification.argtypes = _string_args(4)
    br.br_eval_clustering.argtypes = _string_args(2)
    br.br_eval_detection.argtypes = _string_args(3)
    br.br_eval_detection.restype = c_float
    br.br_eval_landmarking.argtypes = _string_args(3) + [c_int, c_int]
    br.br_eval_landmarking.restype = c_float
    br.br_eval_regression.argtypes = _string_args(4)
    br.br_fuse.argtypes = _var_string_args(3)
    br.br_initialize.argtypes = _var_string_args(1)
    br.br_is_classifier.argtypes = [c_char_p]
    br.br_is_classifier.restype = c_bool
    br.br_make_mask.argtypes = _string_args(3)
    br.br_make_pairwise_mask.argtypes = _string_args(3)
    br.br_most_recent_message.argtypes = [c_char_p, c_int]
    br.br_most_recent_message.restype = c_int
    func = br.br_most_recent_message.__call__
    br.br_most_recent_message = _handle_string_func(func)
    br.br_objects.argtypes = [c_char_p, c_int] + _string_args(2) + [c_bool]
    br.br_objects.restype = c_int
    func2 = br.br_objects.__call__
    br.br_objects = _handle_string_func(func2)
    br.br_plot.argtypes = plot_args
    br.br_plot.restype = c_bool
    br.br_plot_detection.argtypes = plot_args
    br.br_plot_detection.restype = c_bool
    br.br_plot_landmarking.argtypes = plot_args
    br.br_plot_landmarking.restype = c_bool
    br.br_plot_metadata.argtypes = plot_args
    br.br_plot_metadata.restype = c_bool
    br.br_progress.restype = c_float
    br.br_read_pipe.argtypes = [c_char_p, POINTER(c_int), POINTER(POINTER(c_char_p))]
    br.br_scratch_path.argtypes = [c_char_p, c_int]
    br.br_scratch_path.restype = c_int
    func3 = br.br_scratch_path.__call__
    br.br_scratch_path = _handle_string_func(func3)
    br.br_sdk_path.restype = c_char_p
    br.br_get_header.argtypes = [c_char_p, POINTER(c_char_p), POINTER(c_char_p)]
    br.br_set_header.argtypes = _string_args(3)
    br.br_set_property.argtypes = _string_args(2)
    br.br_time_remaining.restype = c_int
    br.br_train.argtypes = _string_args(2)
    br.br_train_n.argtypes = _var_string_args(1)
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
    func4 = br.br_get_filename.__call__
    br.br_get_filename = _handle_string_func(func4)
    br.br_set_filename.argtypes = [c_void_p, c_char_p]
    br.br_get_metadata_string.argtypes = [c_char_p, c_int, c_void_p, c_char_p]
    br.br_get_metadata_string.restype = c_int
    func5 = br.br_get_metadata_string.__call__
    br.br_get_metadata_string = _handle_string_func(func5)
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
    br.br_add_to_gallery.argtypes = [c_void_p, c_void_p]
    br.br_close_gallery.argtypes = [c_void_p]

    return br
