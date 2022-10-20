import click
import yaml


@click.command()
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
@click.option("--workers/--no-workers", type=bool, default=True)
@click.option(
    "-i", "--init-model", type=click.Path(exists=True, dir_okay=False), default=None
)
def train(train_config, workers, init_model=None):
    from contextnet.configs import TrainConfig
    from contextnet.backbones.dense import DenseNet
    from contextnet.pipelines import build_pipeline, get_request, split_batch

    from funlib.geometry import Coordinate

    import gunpowder as gp
    import daisy

    from sklearn.metrics import f1_score
    from tqdm import tqdm
    import zarr
    import torch
    import numpy as np

    import random

    assert torch.cuda.is_available(), "Cannot train reasonably without cuda!"

    def get_mask(train_config, shape):

        # randomly mask part of the input_embeddings
        if train_config.threshold_skew > 0:
            threshold = max(
                [random.random() for _ in range(train_config.threshold_skew)]
            )
        else:
            threshold = 0
        return torch.cuda.FloatTensor(shape).uniform_() > threshold

    def save_snapshot(
        name, dataset: np.ndarray, offset: Coordinate, voxel_size: Coordinate
    ):
        sample = dataset[0]  # select a sample from batch
        if name not in snapshot_zarr:
            snapshot_dataset = snapshot_zarr.create_dataset(
                name,
                shape=(0, *sample.shape),
                dtype=dataset.dtype,
            )
            snapshot_dataset.attrs["resolution"] = voxel_size
            snapshot_dataset.attrs["offset"] = offset
            snapshot_dataset.attrs["axes"] = ["iteration^"] + ["c^", "z", "y", "x"][
                -len(sample.shape) :
            ]
        else:
            snapshot_dataset = snapshot_zarr[name]
        snapshot_dataset.append(sample.reshape((1, *sample.shape)), axis=0)

    train_config = TrainConfig(**yaml.safe_load(open(train_config, "r").read()))
    scale_config = train_config.scale_config
    model_config = train_config.architecture_config
    data_config = train_config.data_config

    model = DenseNet(
        n_input_channels=model_config.raw_input_channels
        + (
            model_config.n_output_channels
            if not model_config.embeddings
            else model_config.num_embeddings
        ),
        n_output_channels=model_config.n_output_channels,
        num_init_features=model_config.num_init_features,
        num_embeddings=model_config.num_embeddings,
        growth_rate=model_config.growth_rate,
        block_config=model_config.block_config,
        padding=model_config.padding,
        upsample_mode=model_config.upsample_mode,
    ).cuda()

    if init_model is not None:
        weights = torch.load(init_model)
        try:
            model.load_state_dict(weights)
        except RuntimeError as e:
            print(e)

    if train_config.loss_file.exists():
        loss_stats = [
            tuple(float(x) for x in line.strip("[]()\n").split(","))
            for line in train_config.loss_file.open("r").readlines()
        ]
    else:
        loss_stats = []

    if train_config.val_file.exists():
        val_stats = [float(x) for x in train_config.val_file.open("r").readlines()]
    else:
        val_stats = []

    snapshot_zarr = zarr.open(f"{train_config.snapshot_container}")

    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.001, total_iters=train_config.warmup
    )

    torch.backends.cudnn.benchmark = True

    if not train_config.checkpoint_dir.exists():
        train_config.checkpoint_dir.mkdir(parents=True)
    checkpoints = sorted([int(f.name) for f in train_config.checkpoint_dir.iterdir()])
    most_recent = 0 if len(checkpoints) == 0 else checkpoints[-1]
    if most_recent > 0:
        weights = torch.load(train_config.checkpoint_dir / f"{most_recent}")
        try:
            model.load_state_dict(weights)
        except RuntimeError as e:
            print("Warning, could not read from checkpoint")
            print(e)
        print(f"Starting from: {most_recent}")
        loss_stats = loss_stats[:most_recent]
        val_stats = val_stats[:most_recent]
    else:
        print(f"Starting from scratch!")
        loss_stats = []
        val_stats = []

    # get pipeline. Stack to create appropriate batch size, add precache
    pipeline = build_pipeline(
        data_config, scale_config, data_config.gt_voxel_size, lsds=train_config.lsds
    )
    pipeline += gp.Stack(train_config.batch_size)
    if workers:
        pipeline += gp.PreCache(num_workers=train_config.num_workers)
    pipeline += gp.PrintProfilingStats(every=10)

    with gp.build(pipeline):

        for i in tqdm(range(most_recent, train_config.num_iterations)):
            batch_request = get_request(
                train_config.input_shape_voxels * 2,
                scale_config,
                lsds=False,
            )
            (
                (raw_high, raw_low, raw_context),
                (target_high, target_low),
                (weight_high, weight_low),
            ) = split_batch(
                pipeline.request_batch(batch_request),
                scale_config,
                lsds=False,
            )

            # convert raw, target and weight to tensor
            torch_raw_context = torch.unsqueeze(
                torch.from_numpy(raw_context.data).cuda().float(), 1
            )
            torch_raw_low = torch.unsqueeze(
                torch.from_numpy(raw_low.data).cuda().float(), 1
            )
            torch_target_low = (
                torch.from_numpy(target_low.data.astype(np.int8)).cuda().float()
            )
            torch_weight_low = torch.from_numpy(weight_low.data).cuda().float()
            torch_raw_high = torch.unsqueeze(
                torch.from_numpy(raw_high.data).cuda().float(), 1
            )
            torch_target_high = (
                torch.from_numpy(target_high.data.astype(np.int8)).cuda().float()
            )
            torch_weight_high = torch.from_numpy(weight_high.data).cuda().float()

            # STEP 1: PREDICT CONTEXT

            # get input embeddings
            raw_shape = raw_context.spec.roi.shape / raw_context.spec.voxel_size
            null_context = torch.zeros(
                (
                    train_config.batch_size,
                    model_config.num_embeddings,
                    *raw_shape,
                )
            )

            embeddings_context, pred_context = model.forward(
                torch_raw_context.cuda().float(),
                null_context.cuda().float(),
            )

            # STEP 1: PREDICT AT LOW RES
            # center crop embeddings
            upsampled_shape = Coordinate(embeddings_context.shape[2:])
            raw_low_shape = raw_low.spec.roi.shape / raw_low.spec.voxel_size
            context = (upsampled_shape - raw_low_shape) / 2
            embeddings_context = embeddings_context[
                (slice(None), slice(None))
                + tuple(slice(c, c + r) for c, r in zip(context, raw_low_shape))
            ]
            # get input embeddings
            pred_mask_low = get_mask(train_config, embeddings_context.shape)

            embeddings_low, pred_low = model.forward(
                torch_raw_low.cuda().float(),
                embeddings_context.cuda().float() * pred_mask_low,
            )
            element_loss_low = loss_func(pred_low, torch_target_low)
            weighted_loss_low = (element_loss_low * torch_weight_low).mean()

            # STEP 2: PREDICT AT HIGH RES

            # center crop pred_low to fit as input to next scale level
            upsampled_shape = Coordinate(embeddings_low.shape[2:])
            raw_high_shape = raw_high.spec.roi.shape / raw_high.spec.voxel_size
            context = (upsampled_shape - raw_high_shape) / 2
            embeddings_low = embeddings_low[
                (slice(None), slice(None))
                + tuple(slice(c, c + r) for c, r in zip(context, raw_high_shape))
            ]

            pred_mask_high = get_mask(train_config, embeddings_low.shape)

            embeddings_high, pred_high = model.forward(
                torch_raw_high.cuda().float(), embeddings_low * pred_mask_high
            )
            element_loss_high = loss_func(pred_high, torch_target_high)
            weighted_loss_high = (element_loss_high * torch_weight_high).mean()

            loss = weighted_loss_low + weighted_loss_high

            # standard training steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_stats.append((weighted_loss_low.item(), weighted_loss_high.item()))

            if i % train_config.checkpoint_interval == 0:
                torch.save(model.state_dict(), train_config.checkpoint_dir / f"{i}")

            if i % train_config.snapshot_interval == 0:
                with train_config.loss_file.open("w") as f:
                    f.write("\n".join([str(x) for x in loss_stats]))

                for name, dataset, offset, voxel_size in zip(
                    [
                        "raw_high",
                        "raw_low",
                        "raw_context",
                        "target_high",
                        "target_low",
                        "weight_high",
                        "weight_low",
                        "pred_high",
                        "pred_low",
                        "pred_context",
                    ],
                    [
                        raw_high.data,
                        raw_low.data,
                        raw_context.data,
                        target_high.data,
                        target_low.data,
                        weight_high.data,
                        weight_low.data,
                        pred_high.detach().cpu().numpy(),
                        pred_low.detach().cpu().numpy(),
                        pred_context.detach().cpu().numpy(),
                    ],
                    [
                        raw_high.spec.roi.offset,
                        raw_low.spec.roi.offset,
                        raw_context.spec.roi.offset,
                        target_high.spec.roi.offset,
                        target_low.spec.roi.offset,
                        weight_high.spec.roi.offset,
                        weight_low.spec.roi.offset,
                        raw_high.spec.roi.offset,
                        raw_low.spec.roi.offset,
                        raw_context.spec.roi.offset,
                    ],
                    [
                        raw_high.spec.voxel_size,
                        raw_low.spec.voxel_size,
                        raw_context.spec.voxel_size,
                        target_high.spec.voxel_size,
                        target_low.spec.voxel_size,
                        weight_high.spec.voxel_size,
                        weight_low.spec.voxel_size,
                        raw_high.spec.voxel_size / 2,
                        raw_low.spec.voxel_size / 2,
                        raw_context.spec.voxel_size / 2,
                    ],
                ):
                    save_snapshot(name, dataset, offset, voxel_size)

                # keep track in an attribute which iterations have been stored
                snapshot_zarr.attrs["iterations"] = snapshot_zarr.attrs.get(
                    "iterations", list()
                ) + [i]
