# !/bin/bash
# ----------------------------------------------------------------------------------------------------
HEIGHT="1024"               # Base height.
WIDTH="1024"                # Base width.
SAMPLES_PER_PROMPT="4"      # Num of samples to generate per prompt.
NROW="2"                    # Grid images per row.

OUTPUT_DIR="outputs/rebuttal"
JSON_PATH="prompts/example.json"
# ----------------------------------------------------------------------------------------------------
MASK_TYPE=("max_norm") 
# usually："max_norm" "crossmap_32" "selfmap_min_max_per_channel" "selfmap_64"
# [
#    "max_norm", "min_max_norm", "binary", "min_max_per_channel", "decoder_map"
#    "selfmap", "selfmap_min_max_per_channel" "selfmap_64"

# ]

CFG=3.5
STEPS=25
mask_reused_step=12

UNET_CONFIG="configs/realcustom_sigdino_highres.json"
UNET_CHECKPOINT="ckpts/realcustom/RealCustom_highres.pth"
UNET_CHECKPOINT_BASE_MODEL="ckpts/sdxl/unet/sdxl-unet.bin"
# ----------------------------------------------------------------------------------------------------
CLIP1_DIR="ckpts/sdxl/clip-sdxl-1"
CLIP2_DIR="ckpts/sdxl/clip-sdxl-2"
VAE_CONFIG_PATH="ckpts/sdxl/vae/sdxl.json"
VAE_CHECKPOINT_PATH="ckpts/sdxl/vae/sdxl-vae.pth"


echo "Start inference"
python3 inference/inference_batch_image.py \
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
    --seed 0 \
    --mask_scope 0.25 \
    --mask_strategy ${MASK_TYPE[*]} \
    --json_path $JSON_PATH