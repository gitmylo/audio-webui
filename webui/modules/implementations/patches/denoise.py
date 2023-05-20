from denoiser.enhance import *


def enhance_new(args, in_file, out_file, model=None, local_out_dir=None):
    # Load model
    if not model:
        model = pretrained.get_model(args).to(args.device)
    model.eval()

    dset = Audioset([(in_file, None)], with_path=True,
                    sample_rate=model.sample_rate, channels=model.chin, convert=True)
    if dset is None:
        return
    loader = distrib.loader(dset, batch_size=1)

    distrib.barrier()

    with ProcessPoolExecutor(1) as pool:
        iterator = LogProgress(logger, loader, name="Generate enhanced files")
        pendings = []
        for data in iterator:
            # Get batch data
            noisy_signals, filenames = data
            noisy_signals = noisy_signals.to(args.device)

            # Forward
            estimate = get_estimate(model, noisy_signals, args)
            for estimate, noisy, filename in zip(estimate, noisy_signals, filenames):
                write(estimate, out_file, sr=model.sample_rate)

        if pendings:
            print('Waiting for pending jobs...')
            for pending in LogProgress(logger, pendings, updates=5, name="Generate enhanced files"):
                pending.result()
