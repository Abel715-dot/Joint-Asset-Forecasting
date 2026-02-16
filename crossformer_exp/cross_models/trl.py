import tltorch

def tensor_regression_layer(x, input_shape, output_shape):
    trl = tltorch.TRL(input_shape, output_shape, factorization='Tucker').to('cuda:0')
    return trl(x)