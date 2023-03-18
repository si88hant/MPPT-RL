from gym.envs.registration import register

register(
   id='mppt_shaded-v0',
   entry_point='gym_mppt.envs:MpptEnvShaded_0',
)

register(
   id='mppt_shaded-v1',
   entry_point='gym_mppt.envs:MpptEnvShaded_1',
)