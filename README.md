# AMG
Implementation code for the paper: "Towards Memorization-Free Diffusion Models".

## Setup 
The codebase incorporates elements from the [improved DDPM](https://github.com/openai/improved-diffusion) and [Stable Diffusion](https://github.com/CompVis/stable-diffusion) repositories, serving as the foundation for implementing our Anti-Memorization Guidance (AMG) on pretrained diffusion models.

To set up the environment, follow the guidelines provided in the aforementioned links. 

To utilize pretrained models, follow these steps to download and place them in the correct directory:

- For unconditional iDDPM, download the checkpoint `cifar10_uncond_50M_500K.pt` from the [improved-diffusion GitHub repository](https://github.com/openai/improved-diffusion) and place it in the `amg/improved-diffusion/models` directory.
- For class-conditional iDDPM, as no pretrained model is provided by the official repository, you should train the model on CIFAR-10 for 500,000 iterations adhering to the [iDDPM's repo](https://github.com/openai/improved-diffusion) default settings. Save the trained model to the `amg/improved-diffusion/models/train500k` directory.
- For text-conditional Stable Diffusion, follow the instructions on its [official repo](https://github.com/CompVis/stable-diffusion) to save `model.ckpt` to the `amg/stable-diffusion/models/ldm/stable-diffusion-v1` directory.
- The pretrained SSCD model `sscd_disc_mixup.torchscript.pt` is available for download at the [sscd-copy-detection GitHub](https://github.com/facebookresearch/sscd-copy-detection) repository. Once downloaded, place it in the `amg/stable-diffusion` directory.

To search nearest neighbors:

- For CIFAR-10, load the data using the command: `python amg/improved-diffusion/scripts/data.py`
- For LAION, no data loading is needed as [`clip-retrieval`](https://github.com/rom1504/clip-retrieval) is utilized for efficient searching of nearest neighbors, which can be installed with the command: `pip install clip-retrieval img2dataset`

## AMG for iDDPM
unconditional generations and evaluations on CIFAR-10
```
python amg/improved-diffusion/scripts/image_sample_guided.py --model_path improved-diffusion/models/cifar10_uncond_50M_500K.pt --image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 --diffusion_steps 4000 --noise_schedule cosine --use_ddim False --num_samples 10000 --timestep_respacing 250 
```

class-conditional generations and evaluations on CIFAR-10
```
python amg/improved-diffusion/scripts/image_sample_guided.py --model_path improved-diffusion/models/train500k/model500000.pt --image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 --diffusion_steps 4000 --noise_schedule cosine --use_ddim False --num_samples 10000 --timestep_respacing 250 --class_cond True 
```

## AMG for Stable Diffusion
text-conditional generations on LAION
```
python amg/stable-diffusion/scripts/txt2img.py --n_iter 1000 --prompt Ann\ Graham\ Lotz --skip_grid --save_npz --guidance_scale 100
```

evaluations on LAION
```
python amg/stable-diffusion/scripts/eval.py
```
