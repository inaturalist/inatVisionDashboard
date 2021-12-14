import hashlib
import os

def path_for_photo_id(base_dir, photo_id):
    msg = str(photo_id).encode('utf-8')
    m = hashlib.sha256()
    m.update(msg)
    hashed = m.hexdigest()
    filename = '{}.jpg'.format(photo_id)
    photo_paths = [base_dir, hashed[0:2], hashed[2:4], filename]
    return os.path.join(*photo_paths)
