import os

from roles import Worker

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if __name__ == '__main__':
    Worker().slave_forever()
