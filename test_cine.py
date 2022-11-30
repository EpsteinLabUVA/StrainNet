import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import os
import scipy.io as sio
from glob import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        config['data_loader']['args']['is_cine'],
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    # logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    os.makedirs(config['save_path'], exist_ok=True)

    with torch.no_grad():
        for i, (data, mask) in enumerate(tqdm(data_loader)):
            data, mask = data.to(device), mask.to(device)
            output = model(data)

            # save test results
            output_cpu = output.cpu().detach().numpy()
            data_cpu = data.cpu().detach().numpy()
            mask_cpu = mask.cpu().detach().numpy()

            path = '{}{}'.format(config['save_path'], str(i+1).zfill(4))
            os.makedirs(path, exist_ok=True)

            sio.savemat('%s/outputs.mat' % path, {"outputs": output_cpu})
            sio.savemat('%s/inputs.mat' % path, {"inputs": data_cpu})


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='StrainNet')
    args.add_argument('-c', '--config', default='config_test_unet_cine.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='./saved/models/StrainNet/0422_000912/model_best.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
