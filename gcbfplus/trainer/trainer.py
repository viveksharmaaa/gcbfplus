import wandb
import os
import numpy as np
import jax

from datetime import datetime
import yaml


import jax.random as jr
import functools as ft

from time import time
from tqdm import tqdm

from gcbfplus.trainer.data import Rollout
from gcbfplus.trainer.utils import rollout
from gcbfplus.env import MultiAgentEnv
from gcbfplus.algo.base import MultiAgentController
from gcbfplus.utils.utils import jax_vmap


class Trainer:

    def __init__(
            self,
            env: MultiAgentEnv,
            env_test: MultiAgentEnv,
            algo: MultiAgentController,
            n_env_train: int,
            n_env_test: int,
            log_dir: str,
            seed: int,
            params: dict,
            save_log: bool = True
    ):
        self.env = env
        self.env_test = env_test
        self.algo = algo
        self.n_env_train = n_env_train
        self.n_env_test = n_env_test
        self.log_dir = log_dir
        self.seed = seed

        if Trainer._check_params(params):
            self.params = params

        # make dir for the models
        if save_log:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            self.model_dir = os.path.join(log_dir, 'models')
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)

        wandb.login()
        wandb.init(name=params['run_name'], project='gcbf+', dir=self.log_dir)


        self.save_log = save_log

        self.steps = params['training_steps']
        self.eval_interval = params['eval_interval']
        self.eval_epi = params['eval_epi']
        self.save_interval = params['save_interval']

        self.update_steps = 0
        self.key = jax.random.PRNGKey(seed)

    @staticmethod
    def _check_params(params: dict) -> bool:
        assert 'run_name' in params, 'run_name not found in params'
        assert 'training_steps' in params, 'training_steps not found in params'
        assert 'eval_interval' in params, 'eval_interval not found in params'
        assert params['eval_interval'] > 0, 'eval_interval must be positive'
        assert 'eval_epi' in params, 'eval_epi not found in params'
        assert params['eval_epi'] >= 1, 'eval_epi must be greater than or equal to 1'
        assert 'save_interval' in params, 'save_interval not found in params'
        assert params['save_interval'] > 0, 'save_interval must be positive'
        return True

    def train(self):
        # record start time
        start_time = time()

        # preprocess the rollout function
        def rollout_fn_single(params, key):
            return rollout(self.env, ft.partial(self.algo.step, params=params), key)

        def rollout_fn(params, keys):
            return jax.vmap(ft.partial(rollout_fn_single, params))(keys)

        rollout_fn = jax.jit(rollout_fn)

        # preprocess the test function
        def test_fn_single(params, key):
            return rollout(self.env_test, lambda graph, k: (self.algo.act(graph, params), None), key)

        def test_fn(params, keys):
            return jax.vmap(ft.partial(test_fn_single, params))(keys)

        test_fn = jax.jit(test_fn)

        #==================================================================
        #Save criteria
        #=======================================================================

        def save_qualified_checkpoint_yaml(
                output_dir: str,
                step: int,
                eval_info: dict,
                qp_eval_info: dict,
                criteria: dict,
                checks: dict,
        ) -> None:
            """
            Save evaluation metrics and checkpoint-selection details to YAML.
            """
            os.makedirs(output_dir, exist_ok=True)

            metadata = {
                "checkpoint": {
                    "step": int(step),
                    "saved_at": datetime.now().astimezone().isoformat(),
                    "reason": "learned_actor_and_qp_teacher_criteria",
                },

                "learned_actor_metrics": {
                    "eval/reward": float(eval_info["eval/reward"]),
                    "eval/reward_final": float(
                        eval_info["eval/reward_final"]
                    ),
                    "eval/cost": float(eval_info["eval/cost"]),
                    "eval/unsafe_frac": float(
                        eval_info["eval/unsafe_frac"]
                    ),
                    "eval/finish": float(eval_info["eval/finish"]),
                    "step": int(eval_info["step"]),
                },

                "qp_teacher_metrics": {
                    "qp_teacher/ever_unsafe": float(
                        qp_eval_info["qp_teacher/ever_unsafe"]
                    ),
                    "qp_teacher/finish_fraction_final": float(
                        qp_eval_info[
                            "qp_teacher/finish_fraction_final"
                        ]
                    ),
                    "qp_teacher/all_finished_final": float(
                        qp_eval_info[
                            "qp_teacher/all_finished_final"
                        ]
                    ),
                    "qp_teacher/all_finished_at_any_time": float(
                        qp_eval_info[
                            "qp_teacher/all_finished_at_any_time"
                        ]
                    ),
                    "qp_teacher/mean_final_goal_error": float(
                        qp_eval_info[
                            "qp_teacher/mean_final_goal_error"
                        ]
                    ),
                    "qp_teacher/max_final_goal_error": float(
                        qp_eval_info[
                            "qp_teacher/max_final_goal_error"
                        ]
                    ),
                    "qp_teacher/minimum_pairwise_distance": float(
                        qp_eval_info[
                            "qp_teacher/minimum_pairwise_distance"
                        ]
                    ),
                    "qp_teacher/maximum_relaxation": float(
                        qp_eval_info[
                            "qp_teacher/maximum_relaxation"
                        ]
                    ),
                    "qp_teacher/positive_maximum_relaxation": max(
                        float(
                            qp_eval_info[
                                "qp_teacher/maximum_relaxation"
                            ]
                        ),
                        0.0,
                    ),
                    "step": int(qp_eval_info["step"]),
                },

                "save_criteria": criteria,

                "criteria_checks": {
                    name: bool(value)
                    for name, value in checks.items()
                },

                "all_criteria_passed": bool(all(checks.values())),
            }

            yaml_path = os.path.join(
                output_dir,
                f"step_{step:04d}_metrics.yaml",
            )

            with open(yaml_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(
                    metadata,
                    file,
                    sort_keys=False,
                    default_flow_style=False,
                )



        #==========================================================================

        # =============================================================
        # Nominal LQR evaluation
        # =============================================================
        def lqr_test_fn_single(key):
            """
            Roll out only the nominal LQR controller u_ref.

            No learned actor and no CBF-QP correction are applied.
            """
            return rollout(
                self.env_test,
                lambda graph, k: (
                    self.env_test.u_ref(graph),
                    None,
                ),
                key,
            )

        def lqr_test_fn(keys):
            return jax.vmap(lqr_test_fn_single)(keys)

        lqr_test_fn = jax.jit(lqr_test_fn)

        # start training
        test_key = jr.PRNGKey(self.seed)
        test_keys = jr.split(test_key, 1_000)[:self.n_env_test]

        # =============================================================
        # Check whether nominal LQR reaches all goals
        # =============================================================
        lqr_rollouts: Rollout = lqr_test_fn(test_keys)

        # Shape:
        # [n_env_test, rollout_horizon, n_agents]
        lqr_finish_fn = jax_vmap(
            jax_vmap(
                self.env_test.finish_mask
            )
        )

        lqr_finish_mask = lqr_finish_fn(
            lqr_rollouts.graph
        )

        lqr_finish_mask_np = np.asarray(
            jax.device_get(lqr_finish_mask)
        ).astype(bool)

        # -------------------------------------------------------------
        # Metric 1:
        # Fraction of individual agents that reach their goal at least
        # once during the rollout.
        # -------------------------------------------------------------
        lqr_agent_ever_finished = np.any(
            lqr_finish_mask_np,
            axis=1,
        )

        lqr_finish_rate = float(
            np.mean(lqr_agent_ever_finished)
        )

        # -------------------------------------------------------------
        # Metric 2:
        # Fraction of environments in which every agent reaches its goal
        # at least once. Agents do not need to finish simultaneously.
        # -------------------------------------------------------------
        lqr_all_agents_finish_rate = float(
            np.mean(
                np.all(
                    lqr_agent_ever_finished,
                    axis=-1,
                )
            )
        )

        # -------------------------------------------------------------
        # Metric 3:
        # Fraction of environments in which all agents are simultaneously
        # inside their goal regions at some point.
        # -------------------------------------------------------------
        lqr_all_simultaneous_rate = float(
            np.mean(
                np.any(
                    np.all(
                        lqr_finish_mask_np,
                        axis=-1,
                    ),
                    axis=-1,
                )
            )
        )

        # -------------------------------------------------------------
        # Metric 4:
        # Fraction of environments in which all agents are inside their
        # goal regions at the final simulation step.
        # -------------------------------------------------------------
        lqr_all_finished_final_rate = float(
            np.mean(
                np.all(
                    lqr_finish_mask_np[:, -1, :],
                    axis=-1,
                )
            )
        )

        # -------------------------------------------------------------
        # First goal-reaching time for each agent.
        # NaN means that the agent never reached its goal.
        # -------------------------------------------------------------
        n_rollouts, rollout_horizon, n_agents = (
            lqr_finish_mask_np.shape
        )

        lqr_first_finish_step = np.full(
            (n_rollouts, n_agents),
            np.nan,
            dtype=np.float32,
        )

        for rollout_idx in range(n_rollouts):
            for agent_idx in range(n_agents):
                reached_steps = np.where(
                    lqr_finish_mask_np[
                        rollout_idx,
                        :,
                        agent_idx,
                    ]
                )[0]

                if reached_steps.size > 0:
                    lqr_first_finish_step[
                        rollout_idx,
                        agent_idx,
                    ] = reached_steps[0]

        lqr_first_finish_time = (
                lqr_first_finish_step
                * float(self.env_test.dt)
        )

        # Mean arrival time over agents that reached their goals.
        if np.any(np.isfinite(lqr_first_finish_time)):
            lqr_mean_arrival_time = float(
                np.nanmean(lqr_first_finish_time)
            )
        else:
            lqr_mean_arrival_time = np.nan

        # -------------------------------------------------------------
        # Optional safety information.
        # The nominal LQR may finish successfully while being unsafe.
        # -------------------------------------------------------------
        lqr_cost = float(
            np.mean(
                lqr_rollouts.costs.sum(axis=-1)
            )
        )

        lqr_unsafe_rate = float(
            np.mean(
                np.asarray(
                    lqr_rollouts.costs.max(axis=-1)
                ) >= 1e-6
            )
        )

        lqr_rollout_duration = (
                rollout_horizon
                * float(self.env_test.dt)
        )

        tqdm.write(
            "\n"
            "============== NOMINAL LQR CHECK =============="
        )

        tqdm.write(
            f"Number of evaluation rollouts: "
            f"{n_rollouts}"
        )

        tqdm.write(
            f"Rollout duration: "
            f"{lqr_rollout_duration:.2f} s"
        )

        tqdm.write(
            f"Individual-agent finish rate: "
            f"{100.0 * lqr_finish_rate:.2f}%"
        )

        tqdm.write(
            f"All-agents finish rate: "
            f"{100.0 * lqr_all_agents_finish_rate:.2f}%"
        )

        tqdm.write(
            f"All-agents simultaneous finish rate: "
            f"{100.0 * lqr_all_simultaneous_rate:.2f}%"
        )

        tqdm.write(
            f"All-agents final-step finish rate: "
            f"{100.0 * lqr_all_finished_final_rate:.2f}%"
        )

        tqdm.write(
            f"Mean first-arrival time: "
            f"{lqr_mean_arrival_time:.3f} s"
        )

        tqdm.write(
            f"Unsafe rollout rate: "
            f"{100.0 * lqr_unsafe_rate:.2f}%"
        )

        tqdm.write(
            f"Mean accumulated cost: "
            f"{lqr_cost:.4f}"
        )

        # Strict test requested by the user.
        lqr_has_100_percent_finish = bool(
            np.isclose(
                lqr_all_agents_finish_rate,
                1.0,
            )
        )

        if lqr_has_100_percent_finish:
            tqdm.write(
                "PASS: LQR achieved a 100% all-agent finish rate."
            )
        else:
            tqdm.write(
                "FAIL: LQR did not achieve a 100% all-agent finish rate."
            )

        tqdm.write(
            "================================================\n"
        )

        wandb.log(
            {
                "lqr/individual_agent_finish_rate":
                    lqr_finish_rate,

                "lqr/all_agents_finish_rate":
                    lqr_all_agents_finish_rate,

                "lqr/all_simultaneous_finish_rate":
                    lqr_all_simultaneous_rate,

                "lqr/all_finished_final_rate":
                    lqr_all_finished_final_rate,

                "lqr/mean_arrival_time":
                    lqr_mean_arrival_time,

                "lqr/unsafe_rate":
                    lqr_unsafe_rate,

                "lqr/cost":
                    lqr_cost,

                "lqr/has_100_percent_finish":
                    float(lqr_has_100_percent_finish),
            },
            step=self.update_steps,
        )

        pbar = tqdm(total=self.steps, ncols=80)
        for step in range(0, self.steps + 1):
            # evaluate the algorithm
            if step % self.eval_interval == 0:
                test_rollouts: Rollout = test_fn(self.algo.actor_params, test_keys)
                total_reward = test_rollouts.rewards.sum(axis=-1) #(n_env_test,batch_size) (num_rollouts, time_horizon=256)
                assert total_reward.shape == (self.n_env_test,)
                reward_min, reward_max = total_reward.min(), total_reward.max()
                reward_mean = np.mean(total_reward)
                reward_final = np.mean(test_rollouts.rewards[:, -1])
                finish_fun = jax_vmap(jax_vmap(self.env_test.finish_mask))
                finish = finish_fun(test_rollouts.graph).max(axis=1).mean()
                cost = test_rollouts.costs.sum(axis=-1).mean()
                unsafe_frac = np.mean(test_rollouts.costs.max(axis=-1) >= 1e-6)
                eval_info = {
                    "eval/reward": reward_mean,
                    "eval/reward_final": reward_final,
                    "eval/cost": cost,
                    "eval/unsafe_frac": unsafe_frac,
                    "eval/finish": finish,
                    "step": step,
                }
                wandb.log(eval_info, step=self.update_steps)
                time_since_start = time() - start_time
                eval_verbose = (f'step: {step:3}, time: {time_since_start:5.0f}s, reward: {reward_mean:9.4f}, '
                                f'min/max reward: {reward_min:7.2f}/{reward_max:7.2f}, cost: {cost:8.4f}, '
                                f'unsafe_frac: {unsafe_frac:6.2f}, finish: {finish:6.2f}')
                tqdm.write(eval_verbose)
  ################################################################################=====================================================
                # ADD THIS BLOCK: evaluate the current QP teacher
                # =====================================================
                if step > 0:
                    qp_key = jr.PRNGKey(
                        self.seed + 100_000 + step
                    )

                    qp_info = (
                        self.algo.evaluate_qp_teacher_training(
                            key=qp_key,
                            rollout_length=self.env._max_step,
                            relax_penalty=1e3,
                        )
                    )

                    qp_eval_info = {
                        "qp_teacher/ever_unsafe":
                            float(qp_info["ever_unsafe"]),

                        "qp_teacher/finish_fraction_final":
                            qp_info["finish_fraction_final"],

                        "qp_teacher/all_finished_final":
                            float(qp_info["all_finished_final"]),

                        "qp_teacher/all_finished_at_any_time":
                            float(
                                qp_info[
                                    "all_finished_at_any_time"
                                ]
                            ),

                        "qp_teacher/mean_final_goal_error":
                            qp_info["mean_final_goal_error"],

                        "qp_teacher/max_final_goal_error":
                            qp_info["max_final_goal_error"],

                        "qp_teacher/minimum_pairwise_distance":
                            qp_info[
                                "minimum_pairwise_distance"
                            ],

                        "qp_teacher/maximum_relaxation":
                            qp_info["maximum_relaxation"],

                        "step": step,
                    }

                    wandb.log(
                        qp_eval_info,
                        step=self.update_steps
                    )

                    qp_verbose = (
                        f"[QP TEACHER] step: {step:3} | "
                        f"unsafe: "
                        f"{qp_info['ever_unsafe']} | "
                        f"finish_frac: "
                        f"{qp_info['finish_fraction_final']:.2f} | "
                        f"all_finished: "
                        f"{qp_info['all_finished_final']} | "
                        f"mean_goal_error: "
                        f"{qp_info['mean_final_goal_error']:.3f} | "
                        f"max_goal_error: "
                        f"{qp_info['max_final_goal_error']:.3f} | "
                        f"min_dist: "
                        f"{qp_info['minimum_pairwise_distance']:.3f} | "
                        f"max_relax: "
                        f"{qp_info['maximum_relaxation']:.3e}"
                    )

                    tqdm.write(qp_verbose)
                    save_criteria = {
                        # Learned actor criteria
                        "eval/max_cost": 0.0,
                        "eval/max_unsafe_frac": 0.0,
                        "eval/min_finish": 0.95,

                        # QP teacher criteria
                        "qp_teacher/require_safe": True,
                        "qp_teacher/min_finish_fraction_final": 1.0,
                        "qp_teacher/require_all_finished_final": True,
                        "qp_teacher/minimum_pairwise_distance": 0.30,
                        "qp_teacher/max_positive_relaxation": 0.15,
                    }

                    positive_qp_relaxation = max(
                        float(
                            qp_eval_info[
                                "qp_teacher/maximum_relaxation"
                            ]
                        ),
                        0.0,
                    )

                    save_checks = {
                        "learned_actor_cost": (
                                float(eval_info["eval/cost"])
                                <= save_criteria["eval/max_cost"]
                        ),

                        "learned_actor_unsafe_frac": (
                                float(eval_info["eval/unsafe_frac"])
                                <= save_criteria["eval/max_unsafe_frac"]
                        ),

                        "learned_actor_finish": (
                                float(eval_info["eval/finish"])
                                >= save_criteria["eval/min_finish"]
                        ),

                        "qp_teacher_safe": (
                                not save_criteria["qp_teacher/require_safe"]
                                or float(
                            qp_eval_info["qp_teacher/ever_unsafe"]
                        ) == 0.0
                        ),

                        "qp_teacher_finish_fraction": (
                                float(
                                    qp_eval_info[
                                        "qp_teacher/finish_fraction_final"
                                    ]
                                )
                                >= save_criteria[
                                    "qp_teacher/min_finish_fraction_final"
                                ]
                        ),

                        "qp_teacher_all_finished": (
                                not save_criteria[
                                    "qp_teacher/require_all_finished_final"
                                ]
                                or float(
                            qp_eval_info[
                                "qp_teacher/all_finished_final"
                            ]
                        ) == 1.0
                        ),

                        "qp_teacher_minimum_distance": (
                                float(
                                    qp_eval_info[
                                        "qp_teacher/minimum_pairwise_distance"
                                    ]
                                )
                                >= save_criteria[
                                    "qp_teacher/minimum_pairwise_distance"
                                ]
                        ),

                        "qp_teacher_relaxation": (
                                positive_qp_relaxation
                                <= save_criteria[
                                    "qp_teacher/max_positive_relaxation"
                                ]
                        ),
                    }

                    criteria_save = all(save_checks.values())
                    periodic_save = step % self.save_interval == 0

                    # periodic_save = step % self.save_interval == 0
                    #
                    # positive_max_relax = max(float(qp_info["maximum_relaxation"]), 0.0)
                    #
                    # criteria_save = (
                    #         float(unsafe_frac) <= 0.0
                    #         and float(cost) <= 0.0
                    #         and float(finish) >= 0.85
                    #         and not bool(qp_info['ever_unsafe'])
                    #         and float(qp_info['finish_fraction_final']) >= 1.0
                    #         and bool(qp_info['all_finished_final'])
                    #         and float(qp_info['minimum_pairwise_distance']) >= 0.30
                    #         and positive_max_relax <= 1e-3
                    # )
#########################################short horizon causing reachnability problems###########################################################
                    # if step > 0 and step % 20 == 0:
                    #     qp_key = jr.PRNGKey(
                    #         self.seed + 100_000 + step
                    #     )
                    #
                    #     for qp_horizon in [256, 512, 768, 1024]:
                    #         qp_info = (
                    #             self.algo.evaluate_qp_teacher_training(
                    #                 key=qp_key,
                    #                 rollout_length=qp_horizon,
                    #                 relax_penalty=1e3,
                    #             )
                    #         )
                    #
                    #         tqdm.write(
                    #             f"[QP HORIZON TEST] "
                    #             f"step={step}, "
                    #             f"horizon={qp_horizon}, "
                    #             f"time={qp_horizon * self.env_test.dt:.2f}s, "
                    #             f"finish_frac="
                    #             f"{qp_info['finish_fraction_final']:.2f}, "
                    #             f"mean_goal_error="
                    #             f"{qp_info['mean_final_goal_error']:.3f}, "
                    #             f"unsafe="
                    #             f"{qp_info['ever_unsafe']}"
                    #         )
# ###########################################################################################################
#                 if self.save_log and step % self.save_interval == 0:
#                     self.algo.save(os.path.join(self.model_dir), step)
#=================================================================================================================
                    if self.save_log:
                        # Existing periodic checkpoint.
                        if periodic_save:
                            self.algo.save(
                                os.path.join(self.model_dir),
                                step,
                            )

                            tqdm.write(
                                f"[PERIODIC CHECKPOINT SAVED] step={step}"
                            )

                        # Save criteria-qualified checkpoint separately.
                        if criteria_save:
                            qualified_dir = os.path.join(
                                self.model_dir,
                                "qualified",
                            )
                            os.makedirs(qualified_dir, exist_ok=True)

                            self.algo.save(
                                qualified_dir,
                                step,
                            )

                            save_qualified_checkpoint_yaml(
                                output_dir=qualified_dir,
                                step=step,
                                eval_info=eval_info,
                                qp_eval_info=qp_eval_info,
                                criteria=save_criteria,
                                checks=save_checks,
                            )

                            tqdm.write(
                                f"[QUALIFIED CHECKPOINT SAVED] "
                                f"step={step} | "
                                f"learned_finish="
                                f"{eval_info['eval/finish']:.2f} | "
                                f"learned_unsafe="
                                f"{eval_info['eval/unsafe_frac']:.2f} | "
                                f"qp_safe="
                                f"{not bool(qp_info['ever_unsafe'])} | "
                                f"qp_finish="
                                f"{qp_info['finish_fraction_final']:.2f} | "
                                f"qp_min_dist="
                                f"{qp_info['minimum_pairwise_distance']:.3f} | "
                                f"qp_relax="
                                f"{positive_qp_relaxation:.3e}"
                            )
#=========================================================================================================================================
            # collect rollouts
            key_x0, self.key = jax.random.split(self.key)
            key_x0 = jax.random.split(key_x0, self.n_env_train)
            rollouts: Rollout = rollout_fn(self.algo.actor_params, key_x0)

            # update the algorithm
            update_info = self.algo.update(rollouts, step)
            wandb.log(update_info, step=self.update_steps)
            self.update_steps += 1

            pbar.update(1)
