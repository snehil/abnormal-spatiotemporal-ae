import logging
import datetime
import os
import sys
import coloredlogs
from classifier import test


device      = 'cpu'
dataset     = 'avenue'
job_uuid    = '603213fe-3308-41d7-8ce2-d734ea4a547b'
epoch       = 2
val_loss    = 0.001326
time_length = 4

job_folder  = os.path.join('./data/clean/{}/jobs'.format(dataset), job_uuid)
log_path    = os.path.join(job_folder, 'logs')

os.makedirs(log_path, exist_ok=True)

logging.basicConfig(filename = os.path.join(log_path, "test-{}.log".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))),
                    level    = logging.DEBUG,
                    format   = "%(asctime)s [%(levelname)s] %(message)s")
coloredlogs.install()
logger = logging.getLogger()


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logger.warning("Ctrl + C triggered by user, testing ended prematurely")
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

if device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    logger.debug("Using CPU only")
elif device == 'gpu0':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    logger.debug("Using GPU 0")
elif device == 'gpu1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    logger.debug("Using GPU 1")
elif device == 'gpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    logger.debug("Using GPU 0 and 1")

test(logger=logger, dataset=dataset, t=time_length, job_uuid=job_uuid, epoch=epoch, val_loss=val_loss,
     visualize_score=True, visualize_frame=False)

logger.info("Job {} ({}) has finished testing.".format(job_uuid, dataset))
