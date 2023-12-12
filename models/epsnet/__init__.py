from .dualenc import DualEncoderEpsNetwork

def get_model(config, lr_init):
    if config.network == 'dualenc':
        return DualEncoderEpsNetwork(config, lr_init)
    else:
        raise NotImplementedError('Unknown network: %s' % config.network)
