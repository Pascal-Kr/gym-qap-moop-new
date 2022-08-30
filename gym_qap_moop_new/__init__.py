from gym.envs.registration import register

name = 'gym_qap_moop'
register(id='QAPMOOP-v0',
         entry_point='gym_qap_moop_new.envs:QAPMOOPenv')
