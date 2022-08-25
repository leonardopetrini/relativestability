def filtering_args(a, args):
    return 1
    # return a['stride'] == 1 and a['width'] <= 64 and 'batch_norm' in a and a['epochs'] == 200
    # return a['batch_size'] == 128
    # # return 1
    # flag = (a['width'] <= 64)
    # return flag if 'batch_norm' in a else 0

# def filtering_args(a, args):
#     if 'background_noise' in a:
#         flag1 = a['background_noise'] == args['background_noise']
#     else:
#         flag1 = 1
#     if 'labelling' in a:
#         flag3 = a['labelling'] == args['labelling']
#     else:
#         flag3 = 1
#     # if 'gaussian_corruption_std' in a:
#     #     flag3 = a['gaussian_corruption_std'] == args['gaussian_corruption_std']
#     # else:
#     #     flag3 = 1
#
#     if 'd' in a:
#         flag2 = a['d'] == args['d']
#     else:
#         flag2 = 1
#     return flag1 * flag2 * flag3
#
#     # return a['width'] == 30 # and a['weight_decay'] == 0.001
#     # if 'bias' in a:
#     #     return a['bias'] == 0
#     # else:
#     #     return 0

def filtering_data(data):
    return 1
    # return data['train loss'][-1] < 0.1