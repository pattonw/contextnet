import click


@click.command()
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
@click.option("--workers/--no-workers", type=bool, default=True)
def train(train_config, workers):
    from contextnet.configs import TrainConfig
    from contextnet.backbones.dense import DenseNet
    from contextnet.pipeline import build_pipeline, get_request, split_batch

    from funlib.geometry import Coordinate

    import gunpowder as gp
    import daisy

    from sklearn.metrics import f1_score
    from tqdm import tqdm
    import zarr
    import torch
    import numpy as np
    import yaml

    import random
    import math

    assert torch.cuda.is_available(), "Cannot train reasonably without cuda!"

    def get_mem_usage():
        return convert_size(torch.cuda.max_memory_allocated(device="cuda"))

    def convert_size(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

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
        padding=model_config.padding,
        upsample_mode=model_config.upsample_mode,
    ).cuda()

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

    loss_func = torch.nn.BCEWithLogitsLoss(reduction="none")
    lsd_loss_func = torch.nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.001, total_iters=train_config.warmup
    )

    torch.backends.cudnn.benchmark = True

    if train_config.start is not None:
        weights = torch.load(train_config.start)
        try:
            model.load_state_dict(weights)
        except RuntimeError as e:
            print(e)

    if not train_config.checkpoint_dir.exists():
        train_config.checkpoint_dir.mkdir(parents=True)
    checkpoints = sorted([int(f.name) for f in train_config.checkpoint_dir.iterdir()])
    most_recent = 0 if len(checkpoints) == 0 else checkpoints[-1]
    if most_recent > 0:
        weights = torch.load(train_config.checkpoint_dir / f"{most_recent}")
        try:
            model.load_state_dict(weights)
        except RuntimeError as e:
            print(e)
            print("Continuing anyway")
        print(f"Starting from: {most_recent}")
        loss_stats = loss_stats[:most_recent]
        val_stats = val_stats[:most_recent]
    else:
        print(f"Starting from scratch!")
        loss_stats = []
        val_stats = []

    # paths
    validation_pred_dataset = "volumes/val/{crop}/{i}/pred/scale_{scale}"
    validation_emb_dataset = "volumes/val/{crop}/{i}/emb/scale_{scale}"
    validation_raw_dataset = "volumes/val/{crop}/raw/scale_{scale}"

    # get pipeline. Stack to create appropriate batch size, add precache
    pipeline = build_pipeline(
        data_config,
        scale_config,
        lsds=train_config.lsds,
        sample_voxel_sizes=train_config.sample_voxel_size,
        use_organelle_datasets=train_config.use_organelle_vols,
    )
    pipeline += gp.Stack(train_config.batch_size)
    if workers:
        pipeline += gp.PreCache(num_workers=train_config.num_workers)

    with gp.build(pipeline):

        for i in tqdm(range(most_recent, train_config.num_iterations)):
            batch_request = get_request(
                train_config.input_shape_voxels * scale_config.scale_factor,
                scale_config,
                lsds=train_config.lsds,
            )
            if not train_config.lsds:
                raws, targets, weights, masks, scale = split_batch(
                    pipeline.request_batch(batch_request),
                    scale_config,
                    lsds=train_config.lsds,
                )
            else:
                (
                    raws,
                    targets,
                    weights,
                    masks,
                    lsd_targets,
                    lsd_masks,
                    scale,
                ) = split_batch(
                    pipeline.request_batch(batch_request),
                    scale_config,
                    lsds=train_config.lsds,
                )

            optimizer.zero_grad()

            # forward pass
            base_raw = raws[-1]
            raw_shape = base_raw.spec.roi.shape / base_raw.spec.voxel_size
            previous_embeddings, previous_pred = (
                torch.zeros(
                    (
                        train_config.batch_size,
                        model_config.num_embeddings,
                        *raw_shape,
                    )
                )
                .cuda()
                .float(),
                torch.zeros(
                    (
                        train_config.batch_size,
                        model_config.n_output_channels,
                        *raw_shape,
                    )
                )
                .cuda()
                .float(),
            )

            losses = []
            weighted_losses = []
            predictions_list = []
            embeddings_list = []
            for scale_level, raw in list(enumerate(raws))[::-1]:
                raw_shape = raw.spec.roi.shape / raw.spec.voxel_size
                # convert raw to tensor and add channel dim
                torch_raw = torch.unsqueeze(
                    torch.from_numpy(raw.data).cuda().float(), 1
                )

                if not model_config.embeddings:
                    previous_pred = torch.nn.Softmax(dim=1)(
                        previous_pred.cuda().float()
                    )
                else:
                    previous_pred = previous_embeddings.cuda().float()

                if Coordinate(previous_pred.shape[2:]) - Coordinate(
                    torch_raw.shape[2:]
                ) != Coordinate(0, 0, 0):
                    upsampled_shape = Coordinate(previous_pred.shape[2:])
                    context = (upsampled_shape - raw_shape) / 2
                    previous_pred = previous_pred[
                        (slice(None), slice(None))
                        + tuple(slice(c, c + r) for c, r in zip(context, raw_shape))
                    ]

                if train_config.threshold_skew > 0:
                    threshold = max(
                        [random.random() for _ in range(train_config.threshold_skew)]
                    )
                else:
                    threshold = 0
                pred_mask = (
                    torch.cuda.FloatTensor(*previous_pred.shape).uniform_() > threshold
                )

                # print(f"Scale level {scale_level}, mem usage: {get_mem_usage()}")
                if train_config.scale_as_input:
                    torch_scale = (
                        torch.from_numpy(np.log2(scale.data))
                        .cuda()
                        .float()
                        .reshape(torch_raw.shape[0], *(1,) * (len(torch_raw.shape) - 1))
                    )
                    scale_array = torch.ones_like(torch_raw) * (torch_scale + scale_level)
                    torch_raw_input = torch.cat([torch_raw, scale_array], 1)
                else:
                    torch_raw_input = torch_raw
                embeddings, pred = model.forward(
                    torch_raw_input.cuda().float(),
                    previous_pred.cuda().float() * pred_mask,
                )
                # print(f"Scale level {scale_level} post forward, mem usage: {get_mem_usage()}")
                previous_embeddings, previous_pred = (embeddings, pred)

                if scale_level < len(targets):
                    if not train_config.lsds:
                        (target, weight, mask) = list(zip(targets, weights, masks))[
                            scale_level
                        ]
                        # convert raw to tensor and add batch dim
                        torch_target = (
                            torch.from_numpy(target.data.astype(np.int8)).cuda().float()
                        )
                        torch_weight = (
                            torch.from_numpy(weight.data * mask.data).cuda().float()
                        )
                        element_loss = loss_func(pred, torch_target)
                        weighted_loss = element_loss * torch_weight

                        weighted_losses.append(weighted_loss)
                        loss = weighted_loss.mean()
                        losses.append(loss)

                    else:
                        (target, weight, mask, lsd_target, lsd_mask,) = list(
                            zip(targets, weights, masks, lsd_targets, lsd_masks)
                        )[scale_level]
                        torch_target = (
                            torch.from_numpy(target.data.astype(np.int8)).cuda().long()
                        )
                        torch_weight = (
                            torch.from_numpy(weight.data * mask.data).cuda().float()
                        )
                        torch_lsd_target = (
                            torch.from_numpy(lsd_target.data).cuda().float()
                        )
                        torch_lsd_mask = torch.from_numpy(lsd_mask.data).cuda().float()
                        element_loss = loss_func(pred[:, :-10], torch_target)
                        lsd_loss = lsd_loss_func(pred[:, -10:], torch_lsd_target)
                        weighted_loss = element_loss * torch_weight
                        weighted_lsd_loss = lsd_loss * torch_lsd_mask

                        weighted_losses.append(weighted_loss)
                        loss = weighted_loss.mean() + weighted_lsd_loss.mean()
                        losses.append(loss)
                predictions_list.append(pred)
                embeddings_list.append(embeddings)

            var, mean = torch.var_mean(pred)
            # print(f"var: {var.item()}, mean: {mean.item()}")
            # print(f"losses: {[loss.item() for loss in losses]}")

            loss = sum(losses) / len(losses)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_stats.append([l.item() for l in losses])

            if i % train_config.checkpoint_interval == 0:
                torch.save(model.state_dict(), train_config.checkpoint_dir / f"{i}")

            if i % train_config.snapshot_interval == 0:
                with train_config.loss_file.open("w") as f:
                    f.write("\n".join([str(x) for x in loss_stats]))

                snapshot_zarr.attrs["iterations"] = snapshot_zarr.attrs.get(
                    "iterations", list()
                ) + [i]
                for scale_level, raw in enumerate(raws):
                    dataset_name = f"raw_s{scale_level}"
                    sample = raw.data[0]  # select a sample from batch
                    if dataset_name not in snapshot_zarr:
                        snapshot_raw = snapshot_zarr.create_dataset(
                            dataset_name,
                            shape=(0, *sample.shape),
                            dtype=raw.data.dtype,
                        )
                        snapshot_raw.attrs["resolution"] = raw.spec.voxel_size
                        snapshot_raw.attrs["offset"] = raw.spec.roi.offset
                        snapshot_raw.attrs["axes"] = ["iteration^", "z", "y", "x"]
                    else:
                        snapshot_raw = snapshot_zarr[dataset_name]
                    snapshot_raw.append(sample.reshape((1, *sample.shape)), axis=0)
                for scale_level, target in enumerate(targets):
                    dataset_name = f"target_s{scale_level}"
                    sample = target.data[0]
                    if dataset_name not in snapshot_zarr:
                        snapshot_target = snapshot_zarr.create_dataset(
                            dataset_name,
                            shape=(0, *sample.shape),
                            dtype=sample.dtype,
                        )
                        snapshot_target.attrs["resolution"] = target.spec.voxel_size
                        snapshot_target.attrs["offset"] = target.spec.roi.offset
                        snapshot_target.attrs["axes"] = [
                            "iteration^",
                            "c^",
                            "z",
                            "y",
                            "x",
                        ]
                    else:
                        snapshot_target = snapshot_zarr[dataset_name]
                    snapshot_target.append(sample.reshape((1, *sample.shape)), axis=0)
                for scale_level, prediction in enumerate(predictions_list[::-1]):
                    sample = prediction.detach().cpu().numpy()[0]
                    dataset_name = f"pred_s{scale_level}"
                    if dataset_name not in snapshot_zarr:
                        snapshot_pred = snapshot_zarr.create_dataset(
                            dataset_name,
                            shape=(0, *sample.shape),
                            dtype=sample.dtype,
                        )
                        snapshot_pred.attrs["resolution"] = (
                            raws[scale_level].spec.voxel_size / 2
                        )
                        snapshot_pred.attrs["offset"] = raws[
                            scale_level
                        ].spec.roi.offset
                        snapshot_pred.attrs["axes"] = [
                            "iteration^",
                            "c^",
                            "z",
                            "y",
                            "x",
                        ]
                    else:
                        snapshot_pred = snapshot_zarr[dataset_name]
                    snapshot_pred.append(sample.reshape((1, *sample.shape)), axis=0)
                for scale_level, embedding in enumerate(embeddings_list[::-1]):
                    sample = embedding.detach().cpu().numpy()[0]
                    dataset_name = f"emb_s{scale_level}"
                    if dataset_name not in snapshot_zarr:
                        snapshot_pred = snapshot_zarr.create_dataset(
                            dataset_name,
                            shape=(0, *sample.shape),
                            dtype=sample.dtype,
                        )
                        snapshot_pred.attrs["resolution"] = (
                            raws[scale_level].spec.voxel_size / 2
                        )
                        snapshot_pred.attrs["offset"] = raws[
                            scale_level
                        ].spec.roi.offset
                        snapshot_pred.attrs["axes"] = [
                            "iteration^",
                            "c^",
                            "z",
                            "y",
                            "x",
                        ]
                    else:
                        snapshot_pred = snapshot_zarr[dataset_name]
                    snapshot_pred.append(sample.reshape((1, *sample.shape)), axis=0)
                for scale_level, weight in enumerate(weights):
                    dataset_name = f"weight_s{scale_level}"
                    sample = weight.data[0]
                    if dataset_name not in snapshot_zarr:
                        snapshot_weight = snapshot_zarr.create_dataset(
                            dataset_name,
                            shape=(0, *sample.shape),
                            dtype=sample.dtype,
                        )
                        snapshot_weight.attrs["resolution"] = weight.spec.voxel_size
                        snapshot_weight.attrs["offset"] = weight.spec.roi.offset
                        snapshot_weight.attrs["axes"] = ["iteration^", "z", "y", "x"]
                    else:
                        snapshot_weight = snapshot_zarr[dataset_name]
                    snapshot_weight.append(sample.reshape((1, *sample.shape)), axis=0)
                for scale_level, weighted_loss in enumerate(weighted_losses[::-1]):
                    sample = weighted_loss.detach().cpu().numpy()[0]
                    dataset_name = f"loss_s{scale_level}"
                    if dataset_name not in snapshot_zarr:
                        snapshot_loss = snapshot_zarr.create_dataset(
                            dataset_name,
                            shape=(0, *sample.shape),
                            dtype=sample.dtype,
                        )
                        snapshot_loss.attrs["resolution"] = targets[
                            scale_level
                        ].spec.voxel_size
                        snapshot_loss.attrs["offset"] = targets[
                            scale_level
                        ].spec.roi.offset
                        snapshot_loss.attrs["axes"] = ["iteration^", "z", "y", "x"]
                    else:
                        snapshot_loss = snapshot_zarr[dataset_name]
                    snapshot_loss.append(sample.reshape((1, *sample.shape)), axis=0)

            if (
                train_config.validation_interval > 0
                and i % train_config.validation_interval == 0
            ):
                model = model.eval()
                # validate
                with torch.no_grad():
                    for dataset_ind, dataset_config in enumerate(data_config.datasets):
                        torch.cuda.empty_cache()
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
                                    dataset_config.gt_dataset.format(
                                        crop_num=validation_crop
                                    ),
                                )
                            except FileExistsError:
                                gt_ds = daisy.open_ds(
                                    dataset_config.fallback_dataset_container,
                                    dataset_config.gt_dataset.format(
                                        crop_num=validation_crop
                                    ),
                                )
                            gt_voxel_size = gt_ds.voxel_size

                            # prepare an empty dataset from which we can pull 0's
                            # in a consistent manner
                            daisy.prepare_ds(
                                str(train_config.validation_container),
                                validation_pred_dataset.format(
                                    i=i,
                                    crop=validation_crop,
                                    scale=scale_config.num_raw_scale_levels,
                                ),
                                total_roi=gt_ds.roi.snap_to_grid(
                                    gt_voxel_size
                                    * 2 ** (scale_config.num_raw_scale_levels),
                                    mode="grow",
                                ),
                                voxel_size=gt_voxel_size
                                * 2 ** (scale_config.num_raw_scale_levels),
                                dtype=np.float32,
                                num_channels=model_config.n_output_channels,
                                delete=True,
                            )
                            daisy.prepare_ds(
                                str(train_config.validation_container),
                                validation_emb_dataset.format(
                                    i=i,
                                    crop=validation_crop,
                                    scale=scale_config.num_raw_scale_levels,
                                ),
                                total_roi=gt_ds.roi.snap_to_grid(
                                    gt_voxel_size
                                    * 2 ** (scale_config.num_raw_scale_levels),
                                    mode="grow",
                                ),
                                voxel_size=gt_voxel_size
                                * 2 ** (scale_config.num_raw_scale_levels),
                                dtype=np.float32,
                                num_channels=model_config.num_embeddings,
                                delete=True,
                            )
                            for scale_level in range(
                                scale_config.num_raw_scale_levels - 1, -1, -1
                            ):
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
                                input_size = (
                                    train_config.eval_input_shape_voxels
                                    * raw_voxel_size
                                )
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
                                out_roi = gt_ds.roi.snap_to_grid(
                                    raw_voxel_size, mode="grow"
                                )
                                if any(
                                    [a < b for a, b in zip(out_roi.shape, input_size)]
                                ):
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
                                    out_roi = out_roi.grow(
                                        context, context
                                    ).snap_to_grid(raw_voxel_size, mode="grow")

                                out_offset = Coordinate(
                                    max(a, b)
                                    for a, b in zip(out_roi.offset, raw_ds.roi.offset)
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
                                                    i=i,
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
                                                i=i,
                                                crop=validation_crop,
                                                scale=scale_level,
                                            ),
                                            emb_key: validation_emb_dataset.format(
                                                i=i,
                                                crop=validation_crop,
                                                scale=scale_level,
                                            ),
                                            raw_key: validation_raw_dataset.format(
                                                crop=validation_crop,
                                                scale=scale_level,
                                            ),
                                        },
                                        output_dir=str(
                                            train_config.validation_container.parent
                                        ),
                                        output_filename=train_config.validation_container.name,
                                    )
                                    + gp.Scan(reference=reference_request)
                                )

                                # prepare the dataset to be written to
                                pred_ds = daisy.prepare_ds(
                                    str(train_config.validation_container),
                                    validation_pred_dataset.format(
                                        i=i,
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
                                        i=i,
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
                                f"Iteration: {i}, crop: {validation_crop}, f1_score: {val_score}"
                            )
                            val_stats.append(val_score)
                            with train_config.val_file.open("w") as f:
                                f.write("\n".join([str(x) for x in val_stats]))

                model = model.train()
