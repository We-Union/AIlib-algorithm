import inspect
from CV import transform_to_painting

def check_func_params(f, param : dict) -> bool:
    f_parser = inspect.getargspec(f)
    args = f_parser.args[1:]
    return (set(args) == set(param.keys()))

if __name__ == "__main__":
    ...