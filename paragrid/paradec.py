def parallel(function):
    def wrapper(all_inputs):
        input_length=len(all_inputs)
        input_names = function.__code__.co_varnames[:input_length]
        args = {i:j for i,j in zip(input_names, all_inputs)}
        return function(**args)
    return wrapper
