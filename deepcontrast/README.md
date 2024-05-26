# deep-contrast
Deep learning methods for style transfer between MRI contrasts

PURPOSE: To train a model (s2nu, residual U-Net model) that converts from one contrast to another for MRIs (ex. T1w fat-saturated to T1w non-fat-saturated breast DCE-MRIs).

MOTIVATION: There are times where you may need both T1w fat and non-fat-sat images to answer a research question, but often times one of these series is missing (usually the non-fat-sat). For example, for I-SPY, we have wanted to use Dong Wei's code to successfully segment FGT and compute BPE within the FGT; however, the FGT segmentation doesn't work too well because we don't have non-fat-sat images, and the code is optimized to extract FGT from non-fat-sat images).

DATA: Basser Dataset (over 2000 images with non-fast-sat, pre-contrast, and first post-contrast MRIs neatly organized by Alex Nguyen. If we need other post-contrast phases, we will need to start with the raw data and pull these out....so let's just work with what we have so far).
- 80% training (1600)
- 10% validation (200)
- 10% testing (200)

COMPONENTS:
1. Model - This is the residual U-Net model itself (like the model class in DeepInPy)
2. Translation - This is the class that characterizes the contrast translation step, namely loading the model and defining other properties necessary for training (like the recon class in DeepInPy)
3. Main script - this is the main script you call to train the model 

Sat2NonSat-Unet -> s2nu