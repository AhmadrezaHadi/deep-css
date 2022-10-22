from gym.envs.registration import register
from .deepcss_v0.parameters import Parameters

pa = Parameters()

kwargs = {"pa": pa}

register(
    id='deepcss-v0',
    entry_point='envs.deepcss_v0.environment:Env',
    max_episode_steps=pa.episode_max_length,
    kwargs=kwargs
)
