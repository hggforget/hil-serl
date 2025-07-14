import numpy as np

class Episode:
    
    def __init__(self, data_dict, post_process, data_keys):
        self._data_keys = data_keys
        
        def _recurisive_resolve(data_dict, path=''):
            for k, v in data_dict.items():
                if (path + '/' + k) in self._data_keys:
                    setattr(self, k, v)
                elif isinstance(v, dict):
                    _recurisive_resolve(v, path + '/' + k)
                    
        _recurisive_resolve(data_dict)
        def _check():
            for key in self._data_keys:
                try:
                    getattr(self, key.split('/')[-1])
                except:
                    raise Exception(f'{key} doesn\'t exist')
        _check()    
        post_process(self)
        self.data_dict = data_dict
    
    def __len__(self):
        return self.length