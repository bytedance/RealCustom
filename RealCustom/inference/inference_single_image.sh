# !/bin/bash
# ----------------------------------------------------------------------------------------------------
HEIGHT="1024"               # Base height.
WIDTH="1024"                # Base width.
SAMPLES_PER_PROMPT="4"      # Num of samples to generate per prompt.
NROW="2"                    # Grid images per row.

OUTPUT_DIR="outputs/test"

# ----------------------------------------------------------------------------------------------------
MASK_TYPE=("max_norm") 
# usually："max_norm" "crossmap_32" "selfmap_min_max_per_channel" "selfmap_64"
# [
#    "max_norm", "min_max_norm", "binary", "min_max_per_channel", "decoder_map"
#    "selfmap", "selfmap_min_max_per_channel" "selfmap_64"

# ]

CFG=7.5
STEPS=25
mask_reused_step=12

UNET_CONFIG="configs/realcustom_sigdino_highres.json"
UNET_CHECKPOINT="ckpts/realcustom/RealCustom_0025000_ema_highres.pth"
UNET_CHECKPOINT_BASE_MODEL="ckpts/sdxl/unet/general_v1-3_sdxl_03.pth"
# ----------------------------------------------------------------------------------------------------
CLIP1_DIR="ckpts/sdxl/clip-sdxl-1"
CLIP2_DIR="ckpts/sdxl/clip-sdxl-2"
VAE_CONFIG_PATH="ckpts/sdxl/vae/sdxl.json"
VAE_CHECKPOINT_PATH="ckpts/sdxl/vae/sdxl-vae.pth"


echo "Start inference"
python3 inference/inference_single_image.py \
    --width $WIDTH \
    --height $HEIGHT \
    --samples_per_prompt $SAMPLES_PER_PROMPT \
    --nrow $NROW \
    --sample_steps $STEPS \
    --guidance_weight $CFG \
    --text_encoder_variant \
        $CLIP1_DIR \
        $CLIP2_DIR \
    --unet_config $UNET_CONFIG \
    --unet_checkpoint $UNET_CHECKPOINT \
    --unet_checkpoint_base_model $UNET_CHECKPOINT_BASE_MODEL \
    --vae_config $VAE_CONFIG_PATH \
    --vae_checkpoint $VAE_CHECKPOINT_PATH \
    --output_dir $OUTPUT_DIR \
    --seed 2024 \
    --text_prompt "the figurine is flying in the sky" \
    --image_prompt_path "prompts/figurine.png" \
    --target_phrase "figurine" \
    --mask_scope 0.25 \
    --mask_strategy ${MASK_TYPE[*]} 