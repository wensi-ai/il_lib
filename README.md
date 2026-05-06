# il_library
Imitation Learning Library with Pytorch Lightning

```
python3 serve.py   robot=a1_iiil   task=pnp/base   arch=residual_a1_state   ckpt_path="/scr/karthikd/Documents/IIIL_Dev/il_lib/outputs/2026-04-15/01-56-02/residual_state_a1_PickPlaceTask_20260415-015602/ckpt/step_200000_no_intervention.pth"   +policy_wrapper.base_policy_ckpt_path="/scr/karthikd/Documents/IIIL_Dev/il_lib/outputs/2026-04-10/00-25-45/diffusion_state_a1_PickPlaceTask_20260410-002545/ckpt/step_125000.pth"   policy_wrapper._target_=il_lib.policies.policy_base.ResidualPolicyWrapper   '~policy_wrapper.deployed_action_steps'   +policy_wrapper.base_deployed_action_steps=8
```

```
python3 serve.py   robot=a1_iiil   task=pnp/base   arch=residual_a1_state   ckpt_path="/scr/karthikd/Documents/IIIL_Dev/il_lib/outputs/2026-04-15/16-52-00/residual_state_a1_PickPlaceTask_20260415-165200/ckpt/step_200000.pth"  +policy_wrapper.base_policy_ckpt_path="/scr/karthikd/Documents/IIIL_Dev/il_lib/outputs/2026-04-10/00-25-45/diffusion_state_a1_PickPlaceTask_20260410-002545/ckpt/step_125000.pth"   policy_wrapper._target_=il_lib.policies.policy_base.ResidualPolicyWrapper   '~policy_wrapper.deployed_action_steps'   +policy_wrapper.base_deployed_action_steps=8
```

ckpt path for 40 demo: /scr/karthikd/Documents/IIIL_Dev/il_lib/outputs/2026-04-19/01-32-49/diffusion_state_a1_PickPlaceTask_20260419-013249/ckpt/step_125000-epoch_2499-loss_0.0058.pth