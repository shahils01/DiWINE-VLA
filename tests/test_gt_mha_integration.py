import unittest

import torch

from models.transformer import (
    Attention,
    GTResidualAttention,
    SoftPromptedTransformer,
    TransformerBlock,
)


class GTMHAIntegrationTest(unittest.TestCase):
    def test_mha_remains_default(self):
        model = SoftPromptedTransformer(
            hidden_size=64,
            multi_modal_input_size=32,
            depth=2,
            num_heads=8,
            num_domains=3,
            dim_action=10,
            dim_propio=10,
            len_soft_prompts=2,
            max_len_seq=32,
        )
        self.assertTrue(all(isinstance(block.attn, Attention) for block in model.blocks))

    def test_gt_mha_forward_backward_and_parameter_reduction(self):
        baseline = TransformerBlock(64, 8, attention_type="mha")
        gt_mha = TransformerBlock(
            64,
            8,
            attention_type="gt_mha_residual",
            gt_mha_num_base_heads=4,
            gt_mha_num_generators=8,
        )
        self.assertIsInstance(gt_mha.attn, GTResidualAttention)
        baseline_attention_params = sum(p.numel() for p in baseline.attn.parameters())
        gt_attention_params = sum(p.numel() for p in gt_mha.attn.parameters())
        self.assertLess(gt_attention_params, baseline_attention_params)

        inputs = torch.randn(2, 11, 64, requires_grad=True)
        output = gt_mha(inputs)
        self.assertEqual(output.shape, inputs.shape)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_mha_checkpoint_projection_conversion(self):
        source = TransformerBlock(64, 8, attention_type="mha")
        target = TransformerBlock(
            64,
            8,
            attention_type="gt_mha_residual",
            gt_mha_num_base_heads=4,
            gt_mha_num_generators=8,
        )
        source_state = source.state_dict()
        q_weight = source_state["attn.qkv.weight"].chunk(3, dim=0)[0]
        expected_q = q_weight.reshape(4, 2, 8, 64).mean(dim=1).reshape(32, 64)

        incompatible = target.load_state_dict(source_state, strict=False)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(torch.allclose(target.attn.q_proj.weight, expected_q))
        self.assertTrue(torch.allclose(target.attn.out_proj.weight, source.attn.proj.weight))
        self.assertTrue(
            set(incompatible.missing_keys).issubset(
                {
                    "attn.generators",
                    "attn.theta",
                    "attn.value_generators",
                    "attn.value_theta",
                }
            )
        )

        gt_state = target.state_dict()
        reloaded = TransformerBlock(
            64,
            8,
            attention_type="gt_mha_residual",
            gt_mha_num_base_heads=4,
            gt_mha_num_generators=8,
        )
        reloaded.load_state_dict(gt_state, strict=True)
        self.assertTrue(torch.allclose(reloaded.attn.q_proj.weight, target.attn.q_proj.weight))

    def test_full_action_transformer_gt_path(self):
        model = SoftPromptedTransformer(
            hidden_size=64,
            multi_modal_input_size=32,
            depth=2,
            num_heads=8,
            num_domains=3,
            dim_action=10,
            dim_propio=10,
            dim_time=8,
            len_soft_prompts=2,
            max_len_seq=32,
            attention_type="gt_mha_residual",
            gt_mha_num_base_heads=4,
            gt_mha_num_generators=8,
        )
        output = model(
            domain_id=torch.tensor([0, 2]),
            vlm_features=torch.randn(2, 3, 32),
            aux_visual_inputs=torch.randn(2, 2, 32),
            action_with_noise=torch.randn(2, 4, 10),
            proprio=torch.randn(2, 10),
            t=torch.rand(2),
        )
        self.assertEqual(output.shape, (2, 4, 10))
        output.mean().backward()


if __name__ == "__main__":
    unittest.main()
