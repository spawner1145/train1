from omegaconf import OmegaConf

model_config = OmegaConf.create(
"""
model:
  target: models.sgm.DiffusionEngine
  params:
    scale_factor: 0.13025
    disable_first_stage_autocast: True

    network_config:
      target: models.sgm.UNetModel
      params:
        adm_in_channels: 2816
        num_classes: sequential
        use_checkpoint: True
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [4, 2]
        num_res_blocks: 2
        channel_mult: [1, 2, 4]
        num_head_channels: 64
        use_spatial_transformer: True
        use_linear_in_transformer: True
        transformer_depth: [1, 2, 10]  # note: the first is unused (due to attn_res starting at 2) 32, 16, 8 --> 64, 32, 16
        context_dim: 2048
        spatial_transformer_attn_type: softmax
        legacy: False

    conditioner_config:
      target: models.sgm.GeneralConditioner
      params:
        emb_models:
          # crossattn cond
          - is_trainable: False
            input_key: prompts
            target: models.sgm.encoders.FrozenCLIPEmbedder
            params:
              layer: hidden
              layer_idx: 11
          # crossattn and vector cond
          - is_trainable: False
            input_key: prompts
            target: models.sgm.encoders.FrozenOpenCLIPEmbedder2
            params:
              arch: ViT-bigG-14
              version: laion2b_s39b_b160k
              layer: penultimate
              always_return_pooled: True
              legacy: False
          # vector cond
          - is_trainable: False
            input_key: original_size_as_tuple
            target: models.sgm.encoders.ConcatTimestepEmbedderND
            params:
              outdim: 256  # multiplied by two
          # vector cond
          - is_trainable: False
            input_key: crop_coords_top_left
            target: models.sgm.encoders.ConcatTimestepEmbedderND
            params:
              outdim: 256  # multiplied by two
          # vector cond
          - is_trainable: False
            input_key: target_size_as_tuple
            target: models.sgm.encoders.ConcatTimestepEmbedderND
            params:
              outdim: 256  # multiplied by two

    first_stage_config:
      target: models.sgm.AutoencoderKLInferenceWrapper
      params:
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          attn_type: vanilla
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [1, 2, 4, 4]
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
""")