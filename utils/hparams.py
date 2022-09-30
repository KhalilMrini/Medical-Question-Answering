class HParams():
    _skip_keys = ['populate_arguments', 'set_from_args', 'print', 'to_dict']
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        if not hasattr(self, item):
            raise KeyError(f"Hyperparameter {item} has not been declared yet")
        setattr(self, item, value)

    def to_dict(self):
        res = {}
        for k in dir(self):
            if k.startswith('_') or k in self._skip_keys:
                continue
            res[k] =  self[k]
        return res

    def populate_arguments(self, parser):
        for k in dir(self):
            if k.startswith('_') or k in self._skip_keys:
                continue
            v = self[k]
            k = k.replace('_', '-')
            if type(v) in (int, float, str):
                parser.add_argument(f'--{k}', type=type(v), default=v)
            elif isinstance(v, bool):
                if not v:
                    parser.add_argument(f'--{k}', action='store_true')
                else:
                    parser.add_argument(f'--no-{k}', action='store_false')

    def set_from_args(self, args):
        for k in dir(self):
            if k.startswith('_') or k in self._skip_keys:
                continue
            if hasattr(args, k):
                self[k] = getattr(args, k)
            elif hasattr(args, f'no_{k}'):
                self[k] = getattr(args, f'no_{k}')

    def print(self):
        for k in dir(self):
            if k.startswith('_') or k in self._skip_keys:
                continue
            print(k, repr(self[k]))


def make_hparams():
    return HParams(
        max_len_train=0, # no length limit
        max_len_dev=0, # no length limit

        numpy_seed=42,

        checks_per_epoch=4,

        learning_rate=2e-6,
        learning_rate_warmup_steps=160,
        clip_grad_norm=0., #no clipping
        step_decay=True, # note that disabling step decay is not implemented
        step_decay_factor=0.5,
        step_decay_patience=5,
        max_consecutive_decays=3,

        embed_model='roberta-large-mnli',
        do_lower_case=False,
        model_path=None,
        embed_max_len=256,

        reduced_dim=256,

        epoch_steps=1000,
        num_epochs=100,
        batch_size=1,
        val_batch_size=16,
        d_proj=64,

        weight_scheme=0,

        load_path="",

        # Knowledge Selector
        d_hidden=1024,
        d_qkv=32,
        n_head=8,
        d_positional=512,
        d_feedforward=2048, 
        dropout=0.1, 
        n_layer=8,
        activation="relu", 
        max_words=1024,
        check_every=40,
        residual_connection=False,
        partitioned=True,

        attn_dropout=0.1,
        relu_dropout=0.1,
        residual_dropout=0.1,
        max_depth=100,

        match_nll_coeff=0.01,
        as2_loss_coeff=0.01,

        print_vocabs=False,

        model_name = "hierarchical",
        model_path_base='./models/'
        )