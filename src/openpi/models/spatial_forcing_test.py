import pathlib
import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.models import pi0_config


def _model(spatial_layer):
    config = pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy", pi05=True,
                                  action_horizon=4, max_token_len=8, hist_horizon=2,
                                  spatial_layer=spatial_layer, spatial_target_dim=8)
    return config, config.create(jax.random.key(0))


def _batch(config, patches):
    observation, actions = config.fake_obs(), config.fake_act()
    images = {name: jnp.repeat(image[:, None], config.hist_horizon, axis=1) if image.ndim == 4 else image
              for name, image in observation.images.items()}
    targets = jax.random.normal(jax.random.key(1), (1, patches, 8))
    mask = jnp.arange(patches)[None, :] >= 3   # the letterbox's first rows are padding
    return observation.replace(images=images, spatial_targets=targets, spatial_target_mask=mask), actions


def test_alignment_loss_trains_the_projector_and_leaves_the_action_loss_alone():
    plain_config, plain = _model(None)
    config, model = _model(1)
    patches = 256
    observation, actions = _batch(config, patches)
    rng = jax.random.key(2)
    action_plain = plain.compute_loss(rng, observation, actions, train=True)
    assert plain.compute_losses(rng, observation, actions, train=True)[1] is None
    chunked, align = model.compute_losses(rng, observation, actions, train=True)
    assert np.isfinite(np.asarray(chunked)).all() and np.isfinite(float(align)) and 0.0 <= float(align) <= 2.0
    np.testing.assert_allclose(np.asarray(model.compute_loss(rng, observation, actions, train=True)), np.asarray(chunked))
    assert action_plain.shape == chunked.shape

    graph, params = nnx.split(model, nnx.Param)

    def align_only(params):
        return nnx.merge(graph, params).compute_losses(rng, observation, actions, train=True)[1]

    grads = jax.grad(align_only)(params)
    flat = {"/".join(map(str, path)): leaf for path, leaf in jax.tree_util.tree_leaves_with_path(grads)}
    assert any("spatial_fc1" in key and np.abs(np.asarray(leaf)).sum() > 0 for key, leaf in flat.items())
    assert any("PaliGemma" in key and np.abs(np.asarray(leaf)).sum() > 0 for key, leaf in flat.items()), \
        "the alignment loss reaches the backbone"


def test_spatial_targets_pick_their_own_cameras_jittered_current_frame():
    key = "observation.images.cam_left_wrist"
    with tempfile.TemporaryDirectory() as directory:
        np.save(f"{directory}/features.npy", np.arange(100, dtype=np.float16)[:, None, None] * np.ones((1, 4, 2), np.float16))
        np.save(f"{directory}/token_mask.npy", np.array([False, True, True, False]))
        targets = _transforms.SpatialTargets(directory, key, {0: 0, 1: 50}, window=20)

        def frame(index, episode, offsets):
            return int(targets({"index": index, "episode_index": episode, "hist_offsets": offsets})["spatial_targets"][0, 0])

        assert frame(70, 1, {key: 0}) == 70         # no jitter: the sample's own frame
        assert frame(70, 1, {key: -2}) == 68        # its camera jittered back two frames
        assert frame(70, 1, {key: 0, "observation.images.cam_right_wrist": -2}) == 70, "another camera's jitter"
        assert frame(70, 1, {key: 2}) == 70         # jitter past the end is held at the end
        assert frame(51, 1, {key: -2}) == 50        # held at the episode's first frame, as LeRobot pads
        out = targets({"index": 10, "episode_index": 0, "hist_offsets": {key: 0}})
        assert out["spatial_targets"].dtype == np.float16
        assert out["spatial_target_mask"].tolist() == [False, True, True, False] and "hist_offsets" not in out


def test_temporal_jitter_records_each_cameras_offset():
    keys = ("observation.images.cam_high", "observation.images.cam_left_wrist")
    jitter = _transforms.TemporalJitter((-2, -1, 0, 1, 2), 5, 3, keys, True)
    data = jitter({key: np.arange(11)[:, None] for key in keys})
    for key in keys:
        assert int(data[key][-1, 0]) == int(np.clip(10 + data["hist_offsets"][key], 0, 10))
    assert _transforms.TemporalJitter((0,), 5, 3, (), False)({})["hist_offsets"] == {}


def test_a_store_that_does_not_fit_is_refused():
    import json
    import types

    from openpi.training.data_loader import _check_spatial_targets

    config = pi0_config.Pi0Config(pi05=True, spatial_layer=12, spatial_target_dim=256)
    with tempfile.TemporaryDirectory() as directory:
        meta = {"frames": 100, "target_dim": 256, "camera": "cam_left_wrist"}
        pathlib_path = pathlib.Path(directory, "meta.json")
        pathlib_path.write_text(json.dumps(meta))
        assert _check_spatial_targets(directory, types.SimpleNamespace(total_frames=100), config) == "cam_left_wrist"
        for change, words in (({"frames": 99}, "frames"), ({"target_dim": 128}, "128-d"), ({"camera": "cam_high"}, "cam_high")):
            pathlib_path.write_text(json.dumps({**meta, **change}))
            try:
                _check_spatial_targets(directory, types.SimpleNamespace(total_frames=100), config)
            except ValueError as error:
                assert words in str(error)
            else:
                raise AssertionError(f"a store with {change} was accepted")
