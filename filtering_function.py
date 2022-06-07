def filtering_args(a):
    return 1
    # return a['width'] == 30 # and a['weight_decay'] == 0.001
    # if 'bias' in a:
    #     return a['bias'] == 0
    # else:
    #     return 0

def filtering_data(data):
    return data['train loss'][-1] < 0.1