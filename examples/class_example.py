# Parallel gridsearch
from paragrid import paragrid

def test_func(a,b): # some function to minimize
    if ((2.5 < a < 10.5) & (-5 > b > -6.5) | (0.5 < a < 3) & (-3.5 > b > -4.5)
        | (-3.5 < a < 0.5) & (-3.5 > b > -4.5)):
        constant = 0
    else:
        constant = -100

    return a**2+1.5*(b+2)**2+100+constant

def find_gradient(parameter, results, number_for_mean = 10):
    parameter_sorted = [x for _,x in sorted(zip(results, parameter))]
    return parameter_sorted[:number_for_mean]

if __name__ == "__main__":
    # spaces
    space_func = {'a': 10, 'b': -7.5}

    params = paragrid(model=test_func, space=space_func,
                      target='min', niter=30,
                      func_type = 'func')
    parameter, results = params.gradient_decent(lr = 1)
    param = params.score()


