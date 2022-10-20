import click


@click.command()
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-i", "--iteration", type=int)
def validate_valid(train_config, iteration):
    from contextnet.configs import TrainConfig
    from contextnet.backbones.dense import DenseNet

    from funlib.geometry import Coordinate

    import gunpowder as gp
    import daisy

    import zarr
    import yaml
    from sklearn.metrics import f1_score
    import torch
    import numpy as np
    from tqdm import tqdm

    from pprint import pprint
    import math

    assert torch.cuda.is_available(), "Cannot validate reasonably without cuda!"

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
        )
        + train_config.scale_as_input,
        n_output_channels=model_config.n_output_channels,
        num_init_features=model_config.num_init_features,
        num_embeddings=model_config.num_embeddings,
        growth_rate=model_config.growth_rate,
        block_config=model_config.block_config,
        padding="valid",
        upsample_mode=model_config.upsample_mode,
    ).cuda()

    model = model.eval()

    torch.backends.cudnn.benchmark = True

    checkpoint = train_config.checkpoint_dir / f"{iteration}"
    assert checkpoint.exists()
    weights = torch.load(checkpoint)
    model.load_state_dict(weights)

    # paths
    validation_pred_dataset = "volumes/val/{crop}/{i}/pred/scale_{scale}"
    validation_emb_dataset = "volumes/val/{crop}/{i}/emb/scale_{scale}"
    validation_raw_dataset = "volumes/val/{crop}/raw/scale_{scale}"

    # setup

    for dataset_config in data_config.dataset_configs:
        # TODO: What happens when s0 has different resolutions for different datasets?
        try:
            raw_s0 = daisy.open_ds(
                dataset_config.raw.container.format(dataset=dataset_config.name),
                dataset_config.raw.dataset.format(level=0),
            )
        except FileExistsError:
            raw_s0 = daisy.open_ds(
                dataset_config.raw.fallback.format(dataset=dataset_config.name),
                dataset_config.raw.dataset.format(level=0),
            )

        for validation_crop in dataset_config.validation:
            try:
                try:
                    gt_ds = daisy.open_ds(
                        dataset_config.raw.container.format(
                            dataset=dataset_config.name
                        ),
                        dataset_config.raw.crop.format(
                            crop_num=validation_crop, organelle="all"
                        ),
                    )
                except (FileExistsError, KeyError):
                    gt_ds = daisy.open_ds(
                        dataset_config.raw.fallback.format(dataset=dataset_config.name),
                        dataset_config.raw.crop.format(
                            crop_num=validation_crop, organelle="all"
                        ),
                    )
            except KeyError:
                print(f"skipping crop {validation_crop}")
                continue
            gt_voxel_size = gt_ds.voxel_size
            gt_roi = gt_ds.roi

            s_metas = {}
            output_roi = gt_roi
            for input_scale_level in range(scale_config.num_eval_scale_levels):
                # contextnet always upsamples by a factor of 2
                s_meta = s_metas.setdefault(input_scale_level, dict())
                s_meta["voxel_size"] = gt_voxel_size * 2 ** (input_scale_level + 1)
                s_meta["output_roi"] = output_roi
                context = s_meta["voxel_size"] * model_config.context
                s_meta["input_roi"] = output_roi.grow(context, context).snap_to_grid(
                    s_meta["voxel_size"]
                )
                s_meta["raw_level"] = int(
                    math.log(min(s_meta["voxel_size"] / raw_s0.voxel_size), 2)
                )
                output_roi = s_meta["input_roi"]

            for scale_level, scale_meta in s_metas.items():
                # prepare output arrays for this scale level
                pprint(scale_meta)
                daisy.prepare_ds(
                    str(train_config.validation_container),
                    validation_pred_dataset.format(
                        i=iteration,
                        crop=validation_crop,
                        scale=scale_level,
                    ),
                    total_roi=scale_meta["output_roi"],
                    voxel_size=scale_meta["voxel_size"] / 2,
                    dtype=np.float32,
                    num_channels=model_config.n_output_channels,
                )
                daisy.prepare_ds(
                    str(train_config.validation_container),
                    validation_emb_dataset.format(
                        i=iteration,
                        crop=validation_crop,
                        scale=scale_level,
                    ),
                    total_roi=scale_meta["output_roi"],
                    voxel_size=scale_meta["voxel_size"] / 2,
                    dtype=np.float32,
                    num_channels=model_config.num_embeddings,
                )
                # prepare input array
                try:
                    input_raw_ds = daisy.prepare_ds(
                        str(train_config.validation_container),
                        validation_raw_dataset.format(
                            crop=validation_crop,
                            scale=scale_level,
                        ),
                        total_roi=scale_meta["input_roi"],
                        voxel_size=scale_meta["voxel_size"],
                        dtype=np.float32,
                        num_channels=None,
                    )
                    try:
                        raw_ds = daisy.open_ds(
                            dataset_config.raw.container.format(
                                dataset=dataset_config.name
                            ),
                            dataset_config.raw.dataset.format(
                                level=scale_meta["raw_level"]
                            ),
                        )
                    except FileExistsError:
                        raw_ds = daisy.open_ds(
                            dataset_config.raw.fallback.format(
                                dataset=dataset_config.name
                            ),
                            dataset_config.raw.dataset.format(
                                level=scale_meta["raw_level"]
                            ),
                        )
                    raw_roi = input_raw_ds.roi.intersect(raw_ds.roi)
                    input_raw_ds[raw_roi] = raw_ds.to_ndarray(raw_roi) / 255.0
                except FileExistsError:
                    input_raw_ds = daisy.open_ds(
                        str(train_config.validation_container),
                        validation_raw_dataset.format(
                            crop=validation_crop,
                            scale=scale_level,
                        ),
                    )
                    assert input_raw_ds.roi == scale_meta["input_roi"]

                if scale_level + 1 not in s_metas:
                    daisy.prepare_ds(
                        str(train_config.validation_container),
                        validation_emb_dataset.format(
                            i=iteration,
                            crop=validation_crop,
                            scale=scale_level + 1,
                        ),
                        total_roi=scale_meta["input_roi"],
                        voxel_size=scale_meta["voxel_size"],
                        dtype=np.float32,
                        num_channels=model_config.num_embeddings,
                    )

            input_shape = train_config.eval_input_shape_voxels
            context = model_config.context

            for scale_level in range(scale_config.num_eval_scale_levels - 1, -1, -1):
                scale_meta = s_metas[scale_level]

                raw_ds = daisy.open_ds(
                    str(train_config.validation_container),
                    validation_raw_dataset.format(
                        crop=validation_crop,
                        scale=scale_level,
                    ),
                )
                in_emb_ds = daisy.open_ds(
                    str(train_config.validation_container),
                    validation_emb_dataset.format(
                        i=iteration,
                        crop=validation_crop,
                        scale=scale_level + 1,
                    ),
                )
                pred_ds = daisy.open_ds(
                    str(train_config.validation_container),
                    validation_pred_dataset.format(
                        i=iteration,
                        crop=validation_crop,
                        scale=scale_level,
                    ),
                    "a",
                )
                emb_ds = daisy.open_ds(
                    str(train_config.validation_container),
                    validation_emb_dataset.format(
                        i=iteration,
                        crop=validation_crop,
                        scale=scale_level,
                    ),
                    "a",
                )

                def predict_in_block(block):
                    in_raw = (
                        torch.from_numpy(
                            raw_ds.to_ndarray(block.read_roi, fill_value=0)
                        )
                        .cuda()
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    in_pred = (
                        torch.from_numpy(
                            in_emb_ds.to_ndarray(block.read_roi, fill_value=0)
                        )
                        .cuda()
                        .unsqueeze(0)
                    )
                    if train_config.scale_as_input:
                        torch_scale = (
                            torch.from_numpy(
                                np.log2(min(raw_ds.voxel_size)).reshape(
                                    (1,) * len(in_raw.shape)
                                )
                                + scale_level
                            )
                            .cuda()
                            .float()
                        )
                        scale_array = torch.ones_like(in_raw) * torch_scale
                        in_raw = torch.cat([in_raw, scale_array], 1)
                    emb, pred = model.forward(in_raw, in_pred)
                    emb = daisy.Array(
                        emb.detach().cpu().numpy()[0],
                        roi=block.write_roi,
                        voxel_size=emb_ds.voxel_size,
                    )
                    pred = daisy.Array(
                        pred.detach().cpu().numpy()[0],
                        roi=block.write_roi,
                        voxel_size=emb_ds.voxel_size,
                    )
                    write_roi = block.write_roi.intersect(emb_ds.roi)
                    emb_ds[write_roi] = emb[write_roi]
                    pred_ds[write_roi] = pred[write_roi]

                input_roi = daisy.Roi(
                    input_shape * 0, input_shape * scale_meta["voxel_size"]
                )
                output_roi = input_roi.grow(
                    -scale_meta["voxel_size"] * context,
                    -scale_meta["voxel_size"] * context,
                )

                pred_task = daisy.Task(
                    f"pred_{scale_level}",
                    raw_ds.roi,
                    read_roi=input_roi,
                    write_roi=output_roi,
                    process_function=predict_in_block,
                    fit="overhang",
                )

                scheduler = daisy.Scheduler([pred_task])
                for _ in tqdm(
                    range(scheduler.task_states[pred_task.task_id].total_block_count)
                ):
                    block = scheduler.acquire_block(pred_task.task_id)
                    if block is None:
                        break
                    predict_in_block(block)
                    block.status = daisy.block.BlockStatus.SUCCESS
                    scheduler.release_block(block)

            # compare prediction s0 to gt
            gt_data = gt_ds.to_ndarray(gt_ds.roi)
            label_data = np.zeros_like(gt_data)
            f1_scores = []
            pred_data = pred_ds.to_ndarray(pred_ds.roi)
            for label, label_ids in enumerate(data_config.categories):
                gt_mask = np.isin(gt_data, label_ids)
                pred_mask = pred_data[label] > 0.5

                val_score = f1_score(
                    gt_mask.flatten(),
                    pred_mask.flatten(),
                )
                print(val_score)
                f1_scores.append(val_score.item())

            # save validation scores on zarr file attribute:
            pred_zarr_array = zarr.open(
                str(train_config.validation_container),
                validation_pred_dataset.format(
                    i=iteration,
                    crop=validation_crop,
                    scale=scale_level,
                ),
            )
            pred_zarr_array.attrs["f1_scores"] = f1_scores

            print(
                f"Iteration: {iteration}, crop: {validation_crop}, f1_scores: {f1_scores}"
            )
