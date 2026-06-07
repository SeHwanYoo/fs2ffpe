from .cyclegan           import CycleGANModel
from .pix2pix            import Pix2PixModel
from .autoencoder        import Autoencoder
from .simple_autoencoder import SimpleAutoencoder
from .uvcgan2            import UVCGAN2
from .uvcgan2_deepthaw   import UVCGAN2DeepThaw

CGAN_MODELS = {
    'cyclegan'           : CycleGANModel,
    'pix2pix'            : Pix2PixModel,
    'autoencoder'        : Autoencoder,
    'simple-autoencoder' : SimpleAutoencoder,
    'uvcgan2'            : UVCGAN2,
    'uvcgan2-deepthaw'   : UVCGAN2DeepThaw,
}

def select_model(name, **kwargs):
    if name not in CGAN_MODELS:
        raise ValueError("Unknown model: %s" % name)

    return CGAN_MODELS[name](**kwargs)

def construct_model(savedir, config, is_train, device):
    model = select_model(
        config.model, savedir = savedir, config = config, is_train = is_train,
        device = device, **config.model_args
    )

    return model
