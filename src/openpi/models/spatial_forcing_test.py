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


def test_spatial_targets_pick_the_jittered_current_frame_within_the_episode():
    with tempfile.TemporaryDirectory() as directory:
        np.save(f"{directory}/features.npy", np.arange(100, dtype=np.float16)[:, None, None] * np.ones((1, 4, 2), np.float16))
        np.save(f"{directory}/token_mask.npy", np.array([False, True, True, False]))
        targets = _transforms.SpatialTargets(directory, {0: 0, 1: 50}, window=20)
        frame = lambda index, episode, offset: int(targets({"index": index, "episode_index": episode,
                                                           "hist_offset": offset})["spatial_targets"][0, 0])
        assert frame(70, 1, 0) == 70        # no jitter: the sample's own frame
        assert frame(70, 1, -2) == 68       # jittered back two frames
        assert frame(70, 1, 2) == 70        # jitter past the end is held at the end
        assert frame(51, 1, -2) == 50       # held at the episode's first frame, as LeRobot pads
        out = targets({"index": 10, "episode_index": 0, "hist_offset": 0})
        assert out["spatial_target_mask"].tolist() == [False, True, True, False] and "hist_offset" not in out
