
__all__ = [
    'cells', 'config', 'experiments', 'layers', 'RnO', 'smoothFuncs',
    'testLayers', 'testRnO', 'utils'
]

def reload(rebuild_cache=True):
    """
    Convenience furnction to reload all modules.
    """
    from importlib import import_module, invalidate_caches
    if rebuild_cache:
        print("Rebuilding modules.")
        invalidate_caches()
    for module_name in __all__:
        globals()[module_name] = import_module(f'.{module_name}', __package__)

reload(rebuild_cache=False)
