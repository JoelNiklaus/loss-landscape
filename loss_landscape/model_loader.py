import importlib
import os


def load(dataset, model_name, model_file, data_parallel=False):
    """
    Dynamically invokes the load function of the dataset specified
    Example: cifar10 -> Invoke load function of datasets.cifar10.model_loader
    Args:
        dataset:        the name of the dataset
        model_name:     the name of the model
        model_file:
        data_parallel:

    Returns:
        the loaded model
    """
    available_datasets = [f.name for f in os.scandir("../datasets") if f.is_dir()]
    try:
        available_datasets.remove("__pycache__")
    except:
        print('No pycache found!')
    if dataset not in available_datasets:
        raise ValueError(dataset + " is not a valid dataset. The available datasets are " + str(available_datasets))

    module = 'datasets.' + dataset + '.model_loader'
    mymod = importlib.import_module(module)  # import the module: same as import datasets.{dataset}.model_loader

    load_function = getattr(mymod, "load")
    return load_function(model_name, model_file, data_parallel)
