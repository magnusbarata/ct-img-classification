import os
import shutil
import json

def create_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        usr_in = input('%s already exists. Overwrite? (y/n): ' % path)
        if usr_in.lower() == 'y':
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            usr_in = input('Continue training this model? (y/n): ')
            if usr_in.lower() == 'y': return True
            else:
                print('To evaluate without training, use --eval only. Exiting...')
                raise SystemExit

class Params:
    def __init__(self, fparams):
        self.update(fparams)

    def save(self, fparams):
        with open(fparams, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, fparams):
        with open(fparams) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def dict(self):
        return self.__dict__
