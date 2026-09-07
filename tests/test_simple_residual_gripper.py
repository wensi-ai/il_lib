from types import SimpleNamespace

import torch

from il_lib.policies.policy_base import ResidualPolicyWrapper
from il_lib.policies.simple_residual_policy import SimpleResidualPolicy, _SimpleResidualMLP


def test_action_head_uses_full_scale_for_gripper_only():
    head = _SimpleResidualMLP(
        input_dim=1,
        output_dim=3,
        hidden_dim=1,
        hidden_depth=1,
        use_tanh_output=True,
        output_scale=[0.2, 0.2, 1.0],
    )
    with torch.no_grad():
        head.input_layer.weight.fill_(1.0)
        head.input_layer.bias.zero_()
        head.output_layer.weight.fill_(10.0)
        head.output_layer.bias.zero_()
    output = head(torch.ones(1, 1))
    assert torch.all(output[..., :2].abs() <= 0.2)
    assert output[..., 2].abs() > 0.9


def test_absolute_gripper_target_supports_every_gripper_index():
    policy = object.__new__(SimpleResidualPolicy)
    policy._gripper_action_indices = [1, 3]
    policy._predict_direct_action = False
    policy._learn_gripper_action = True
    policy._gripper_action_mode = "absolute"
    base = torch.tensor([[0.1, 1.0, -0.2, -1.0]])
    oracle = torch.tensor([[0.3, -1.0, -0.1, 1.0]])

    target, base_gripper, oracle_gripper = policy._build_action_target(base, oracle)

    torch.testing.assert_close(target, torch.tensor([[0.2, -1.0, 0.1, 1.0]]))
    torch.testing.assert_close(base_gripper, torch.tensor([[1.0, -1.0]]))
    torch.testing.assert_close(oracle_gripper, torch.tensor([[-1.0, 1.0]]))


def test_wrapper_replaces_gripper_but_adds_arm_residual():
    wrapper = object.__new__(ResidualPolicyWrapper)
    wrapper.robot_type = "A1"
    wrapper._clamp_combined_arm_action = False
    wrapper.policy = SimpleNamespace(
        _predict_direct_action=False,
        _gripper_action_mode="absolute",
    )
    base = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0])
    residual = torch.tensor([0.01, -0.02, 0.0, 0.0, 0.0, 0.0, -0.8])

    combined, logged_residual = wrapper._combine_normalized_actions(base, residual)

    torch.testing.assert_close(combined[:6], base[:6] + residual[:6])
    torch.testing.assert_close(combined[6:], torch.tensor([-0.8]))
    torch.testing.assert_close(logged_residual, residual)


def test_transic_target_normalization_uses_theoretical_joint_delta_range():
    policy = object.__new__(SimpleResidualPolicy)
    policy.action_dim = 3
    policy.action_prediction_horizon = 1
    policy._residual_target_normalization = "transic"
    policy._residual_target_normalization_mask = torch.tensor([True, True, False])

    target = torch.tensor([[0.4, -2.5, 1.0]])
    normalized = policy._normalize_residual_target(target)
    torch.testing.assert_close(normalized, torch.tensor([[0.2, -1.0, 1.0]]))

    restored = policy._denormalize_residual_target(normalized)
    torch.testing.assert_close(restored, torch.tensor([[0.4, -2.0, 1.0]]))


def test_transic_wrapper_clamps_arm_after_composition_but_not_absolute_gripper():
    wrapper = object.__new__(ResidualPolicyWrapper)
    wrapper.robot_type = "A1"
    wrapper._clamp_combined_arm_action = True
    wrapper.policy = SimpleNamespace(
        _predict_direct_action=False,
        _gripper_action_mode="absolute",
    )
    base = torch.tensor([0.95, -0.95, 0.0, 0.0, 0.0, 0.0, 1.0])
    residual = torch.tensor([0.2, -0.2, 0.0, 0.0, 0.0, 0.0, -0.8])

    combined, _ = wrapper._combine_normalized_actions(base, residual)

    torch.testing.assert_close(combined[:2], torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(combined[6:], torch.tensor([-0.8]))


def _ablation_wrapper(**flags):
    wrapper = object.__new__(ResidualPolicyWrapper)
    wrapper.robot_type = "A1"
    wrapper._clamp_combined_arm_action = False
    wrapper._gripper_from_base = flags.get("gripper_from_base", False)
    wrapper._arm_from_base = flags.get("arm_from_base", False)
    wrapper.policy = SimpleNamespace(
        _predict_direct_action=False,
        _gripper_action_mode="absolute",
    )
    return wrapper


def test_wrapper_gripper_from_base_keeps_base_gripper_and_residual_arm():
    wrapper = _ablation_wrapper(gripper_from_base=True)
    base = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0])
    residual = torch.tensor([0.01, -0.02, 0.0, 0.0, 0.0, 0.0, 0.9])

    combined, logged_residual = wrapper._combine_normalized_actions(base, residual)

    torch.testing.assert_close(combined[:6], base[:6] + residual[:6])
    torch.testing.assert_close(combined[6:], torch.tensor([-1.0]))
    # The trace still records what the residual wanted.
    torch.testing.assert_close(logged_residual, residual)


def test_wrapper_arm_from_base_keeps_base_arm_and_residual_gripper():
    wrapper = _ablation_wrapper(arm_from_base=True)
    base = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0])
    residual = torch.tensor([0.01, -0.02, 0.0, 0.0, 0.0, 0.0, 0.9])

    combined, _ = wrapper._combine_normalized_actions(base, residual)

    torch.testing.assert_close(combined[:6], base[:6])
    torch.testing.assert_close(combined[6:], torch.tensor([0.9]))
