
def get_pretraining_set_intra(opts):

    from feeder.feeder_pretraining_intra import Feeder
    training_data = Feeder(**opts.train_feeder_args)

    return training_data

def get_pretraining_set_inter(opts):

    from feeder.feeder_pretraining_inter import Feeder
    # from feeder.feeder_pretrain_hdm05 import Feeder
    training_data = Feeder(**opts.train_feeder_args)

    return training_data


def get_finetune_training_set(opts):

    from feeder.feeder_downstream import Feeder

    data = Feeder(**opts.train_feeder_args)

    return data

def get_finetune_validation_set(opts):

    from feeder.feeder_downstream import Feeder
    data = Feeder(**opts.test_feeder_args)

    return data
def get_hdm05_training_set(opts):

    from feeder.feeder_hdm05_attack import Feeder

    data = Feeder(**opts.train_feeder_args)

    return data

def get_hdm05_validation_set(opts):

    from feeder.feeder_hdm05_attack import Feeder
    data = Feeder(**opts.test_feeder_args)

    return data
def get_attack_training_set(opts):

    from feeder.feeder_attack import Feeder

    data = Feeder(**opts.train_feeder_args)

    return data

def get_attack_validation_set(opts):

    from feeder.feeder_attack import Feeder
    data = Feeder(**opts.test_feeder_args)

    return data

def get_finetune_training_set_semi_supervised(opts):

    from feeder.feeder_downstream_semi_supervised import Feeder

    data = Feeder(**opts.train_feeder_args)

    return data

def get_finetune_validation_set_semi_supervised(opts):

    from feeder.feeder_downstream_semi_supervised import Feeder
    data = Feeder(**opts.test_feeder_args)

    return data
