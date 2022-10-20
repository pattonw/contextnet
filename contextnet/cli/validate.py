import click


@click.command()
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-i", "--iteration", type=int)
def validate(train_config, iteration):
    from contextnet.configs import TrainConfig
    from contextnet.backbones.dense import DenseNet
    
    from funlib.geometry import Coordinate

    import gunpowder as gp
    import daisy

    import yaml
    from sklearn.metrics import f1_score
    import torch
    import numpy as np

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
        ),
        n_output_channels=model_config.n_output_channels,
        num_init_features=model_config.num_init_features,
        num_embeddings=model_config.num_embeddings,
        growth_rate=model_config.growth_rate,
        block_config=model_config.block_config,
        padding=model_config.padding,
        upsample_mode=model_config.upsample_mode,
    ).cuda()

    torch.backends.cudnn.benchmark = True

    checkpoint = train_config.checkpoint_dir / f"{iteration}"
    assert checkpoint.exists()
    weights = torch.load(checkpoint)
    model.load_state_dict(weights)

    # paths
    validation_pred_dataset = "volumes/val/{crop}/{i}/pred/scale_{scale}"
    validation_emb_dataset = "volumes/val/{crop}/{i}/emb/scale_{scale}"
    validation_raw_dataset = "volumes/val/{crop}/raw/scale_{scale}"

    #setup

    model = model.eval()
    # validate
    with torch.no_grad():
        for dataset_config in data_config.datasets:
            # TODO: What happens when s0 has different resolutions for different datasets?
            try:
                raw_s0 = daisy.open_ds(
                    dataset_config.dataset_container,
                    dataset_config.raw_dataset + f"/s0",
                )
            except FileExistsError:
                raw_s0 = daisy.open_ds(
                    dataset_config.fallback_dataset_container,
                    dataset_config.raw_dataset + f"/s0",
                )
            assert (
                raw_s0.voxel_size == data_config.gt_voxel_size * 2
            ), f"gt resolution is not double the raw s0 resolution: raw({raw_s0.voxel_size}):gt({data_config.gt_voxel_size})"
            for validation_crop in dataset_config.validation_crops:
                try:
                    gt_ds = daisy.open_ds(
                        dataset_config.dataset_container,
                        dataset_config.gt_group.format(crop_num=validation_crop)
                        + "/all",
                    )
                except FileExistsError:
                    gt_ds = daisy.open_ds(
                        dataset_config.fallback_dataset_container,
                        dataset_config.gt_group.format(crop_num=validation_crop)
                        + "/all",
                    )
                gt_voxel_size = gt_ds.voxel_size

                # prepare an empty dataset from which we can pull 0's
                # in a consistent manner
                daisy.prepare_ds(
                    str(train_config.validation_container),
                    validation_pred_dataset.format(
                        i=iteration,
                        crop=validation_crop,
                        scale=scale_config.num_raw_scale_levels,
                    ),
                    total_roi=gt_ds.roi.snap_to_grid(
                        gt_voxel_size * 2 ** (scale_config.num_raw_scale_levels),
                        mode="grow",
                    ),
                    voxel_size=gt_voxel_size * 2 ** (scale_config.num_raw_scale_levels),
                    dtype=np.float32,
                    num_channels=model_config.n_output_channels,
                    delete=True,
                )
                daisy.prepare_ds(
                    str(train_config.validation_container),
                    validation_emb_dataset.format(
                        i=iteration,
                        crop=validation_crop,
                        scale=scale_config.num_raw_scale_levels,
                    ),
                    total_roi=gt_ds.roi.snap_to_grid(
                        gt_voxel_size * 2 ** (scale_config.num_raw_scale_levels),
                        mode="grow",
                    ),
                    voxel_size=gt_voxel_size * 2 ** (scale_config.num_raw_scale_levels),
                    dtype=np.float32,
                    num_channels=model_config.num_embeddings,
                    delete=True,
                )
                for scale_level in range(scale_config.num_raw_scale_levels - 1, -1, -1):
                    # assumptions:
                    # 1) raw data is provided as a scale pyramid
                    # 2) gt data is provided in labels/all

                    try:
                        raw_ds = daisy.open_ds(
                            dataset_config.dataset_container,
                            dataset_config.raw_dataset + f"/s{scale_level}",
                        )
                    except FileExistsError:
                        raw_ds = daisy.open_ds(
                            dataset_config.fallback_dataset_container,
                            dataset_config.raw_dataset + f"/s{scale_level}",
                        )
                    raw_voxel_size = raw_ds.voxel_size

                    raw_key, upsampled_key, pred_key, emb_key = (
                        gp.ArrayKey("RAW"),
                        gp.ArrayKey("UPSAMPLED"),
                        gp.ArrayKey("PRED"),
                        gp.ArrayKey("EMB"),
                    )
                    input_size = train_config.eval_input_shape_voxels * raw_voxel_size
                    output_size = input_size
                    reference_request = gp.BatchRequest()
                    reference_request.add(
                        raw_key,
                        input_size,
                    )
                    reference_request.add(
                        upsampled_key,
                        input_size,
                    )
                    reference_request.add(
                        pred_key,
                        output_size,
                    )
                    if model_config.embeddings:
                        reference_request.add(emb_key, output_size)
                    out_voxel_size = raw_voxel_size / 2
                    out_roi = gt_ds.roi.snap_to_grid(raw_voxel_size, mode="grow")
                    if any([a < b for a, b in zip(out_roi.shape, input_size)]):
                        context = (
                            gp.Coordinate(
                                *(
                                    max(out_shape - gt_shape, 0)
                                    for gt_shape, out_shape in zip(
                                        out_roi.shape, output_size
                                    )
                                )
                            )
                            + 1
                        ) / 2
                        out_roi = out_roi.grow(context, context).snap_to_grid(
                            raw_voxel_size, mode="grow"
                        )

                    out_offset = Coordinate(
                        max(a, b) for a, b in zip(out_roi.offset, raw_ds.roi.offset)
                    )
                    out_offset += (-out_offset) % out_voxel_size
                    out_roi.offset = out_offset

                    val_pipeline = (
                        (
                            gp.ZarrSource(
                                str(dataset_config.dataset_container),
                                {raw_key: f"volumes/raw/s{scale_level}"},
                                array_specs={
                                    raw_key: gp.ArraySpec(
                                        roi=raw_ds.roi,
                                        voxel_size=raw_ds.voxel_size,
                                        interpolatable=True,
                                    )
                                },
                            )
                            + gp.Normalize(raw_key),
                            gp.ZarrSource(
                                str(train_config.validation_container),
                                {
                                    upsampled_key: (
                                        validation_pred_dataset
                                        if not model_config.embeddings
                                        else validation_emb_dataset
                                    ).format(
                                        i=iteration,
                                        crop=validation_crop,
                                        scale=scale_level + 1,
                                    )
                                },
                            )
                            + gp.Pad(upsampled_key, None),
                        )
                        + gp.MergeProvider()
                        + gp.Unsqueeze([raw_key])
                        + gp.Unsqueeze([raw_key, upsampled_key])
                        + Predict(
                            model=model,
                            inputs={
                                "raw": raw_key,
                                "upsampled": upsampled_key,
                            },
                            outputs={0: emb_key, 1: pred_key},
                            array_specs={
                                pred_key: gp.ArraySpec(
                                    roi=out_roi,
                                    voxel_size=out_voxel_size,
                                    dtype=np.float32,
                                ),
                                emb_key: gp.ArraySpec(
                                    roi=out_roi,
                                    voxel_size=out_voxel_size,
                                    dtype=np.float32,
                                ),
                            },
                        )
                        + gp.Squeeze([raw_key, emb_key, pred_key])
                        + gp.Squeeze([raw_key])
                        + gp.ZarrWrite(
                            dataset_names={
                                pred_key: validation_pred_dataset.format(
                                    i=iteration,
                                    crop=validation_crop,
                                    scale=scale_level,
                                ),
                                emb_key: validation_emb_dataset.format(
                                    i=iteration,
                                    crop=validation_crop,
                                    scale=scale_level,
                                ),
                                raw_key: validation_raw_dataset.format(
                                    crop=validation_crop,
                                    scale=scale_level,
                                ),
                            },
                            output_dir=str(train_config.validation_container.parent),
                            output_filename=train_config.validation_container.name,
                        )
                        + gp.Scan(reference=reference_request)
                    )

                    # prepare the dataset to be written to
                    pred_ds = daisy.prepare_ds(
                        str(train_config.validation_container),
                        validation_pred_dataset.format(
                            i=iteration,
                            crop=validation_crop,
                            scale=scale_level,
                        ),
                        total_roi=out_roi,
                        voxel_size=out_voxel_size,
                        dtype=np.float32,
                        write_size=output_size,
                        num_channels=model_config.n_output_channels,
                        delete=True,
                    )

                    # prepare emb ds
                    daisy.prepare_ds(
                        str(train_config.validation_container),
                        validation_emb_dataset.format(
                            i=iteration,
                            crop=validation_crop,
                            scale=scale_level,
                        ),
                        total_roi=out_roi,
                        voxel_size=out_voxel_size,
                        dtype=np.float32,
                        write_size=output_size,
                        num_channels=model_config.num_embeddings,
                        delete=True,
                    )
                    # prepare raw ds
                    daisy.prepare_ds(
                        str(train_config.validation_container),
                        validation_raw_dataset.format(
                            crop=validation_crop,
                            scale=scale_level,
                        ),
                        total_roi=out_roi,
                        voxel_size=raw_voxel_size,
                        dtype=np.float32,
                        write_size=output_size,
                        num_channels=None,
                        delete=True,
                    )

                    with gp.build(val_pipeline):
                        val_pipeline.request_batch(gp.BatchRequest())

                # compare prediction s0 to gt
                gt_data = gt_ds.to_ndarray(gt_ds.roi)
                label_data = np.zeros_like(gt_data)
                for label, label_ids in enumerate(data_config.categories):
                    label_data[np.isin(gt_data, label_ids)] = label + 1
                pred_data = pred_ds.to_ndarray(pred_ds.roi)
                pred_data = np.argmax(pred_data, axis=0)

                val_score = f1_score(
                    label_data.flatten(),
                    pred_data.flatten(),
                    average=None,
                )
                print(
                    f"Iteration: {iteration}, crop: {validation_crop}, f1_score: {val_score}"
                )
