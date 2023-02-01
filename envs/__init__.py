from gym.envs.registration import register
from .deepcss_v0.parameters import Parameters as V0_PA
from .deepcss_v1.parameters import Parameters as V1_PA
from .deepcss_v2.parameters import Parameters as V2_PA

v0_pa = V0_PA()
v1_pa = V1_PA()
v2_pa = V2_PA()

v0_kwargs = {"pa": v0_pa}
v1_kwargs = {"pa": v1_pa}
v2_kwargs = {"pa": v2_pa}

# Register deepcss-v0
register(
    id='deepcss-v0',
    entry_point='envs.deepcss_v0.environment:Env',
    max_episode_steps=v0_pa.episode_max_length,
    kwargs=v0_kwargs
)

# Register deepcss-v1
register(
    id='deepcss-v1',
    entry_point='envs.deepcss_v1.environment:Env',
    max_episode_steps=v1_pa.episode_max_length,
    kwargs=v1_kwargs
)

# Register deepcss-v2
register(
    id='deepcss-v2',
    entry_point='envs.deepcss_v2.environment:Env',
    max_episode_steps=v2_pa.episode_max_length,
    kwargs=v2_kwargs
)
