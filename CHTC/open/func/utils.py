import os
import shutil


_log_path = None

def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def time_str(t):
  if t >= 3600:
    return '{:.1f}h, {:.1f}min'.format(t // 3600, (t % 3600)/60) # time in hours
  if t >= 60:
    return '{:.1f}m, {:.1f}s'.format(t // 60, t % 60)   # time in minutes
  return '{:.1f}s'.format(t)