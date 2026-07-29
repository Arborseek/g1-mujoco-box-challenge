import numpy as np
import mujoco

from src.controllers.demo import DEFAULT_LEG, LEGS, RIGHT_ARM, WAIST
from src.env import BoxPickPlaceEnv

BEND_WAIST = np.zeros(3)

env = BoxPickPlaceEnv.create()
waist_ids = [env.model.actuator(n).id for n in WAIST]
arm_ids = [env.model.actuator(n).id for n in RIGHT_ARM]
leg_ids = [env.model.actuator(n).id for n in LEGS]
box = env.body_pos(env.box_body_ids[0])
print("box0", box, "pick", env.initial_box_qpos[0:2] + np.array([0.38, 0]))

rx, ry, rz = -0.50, -0.56, 0.36
for sp in [0.85, 0.95, 1.05]:
    for sr in [0.2, 0.35]:
        for el in [0.55, 0.70, 0.85]:
            env.reset()
            env.data.qpos[0:3] = [rx, ry, rz]
            env.data.qpos[3:7] = [1, 0, 0, 0]
            for i, aid in enumerate(leg_ids):
                env.data.ctrl[aid] = DEFAULT_LEG[i]
            for i, aid in enumerate(waist_ids):
                env.data.ctrl[aid] = BEND_WAIST[i]
            arm = [sp, sr, 0.05, el, 0.0, 0.58, 0.0]
            for i, aid in enumerate(arm_ids):
                env.data.ctrl[aid] = arm[i]
            for _ in range(2000):
                mujoco.mj_step(env.model, env.data)
            d = float(np.linalg.norm(env.hand_pos() - box))
            if d < 0.45:
                print(f"sp={sp} sr={sr} el={el} d={d:.3f}")
