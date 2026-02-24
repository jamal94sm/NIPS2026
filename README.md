NIPS_2026.py : 
finetuning last stage of ConvNeXt foundation model with CASIS-MS-ROI using ArcFace loss function. 

NIPS_2.py: 
finetuning last stage of ConvNeXt foundation model with CASIS-MS-ROI using ArcFace and SupCon (supervised contrastive) loss functions.

NIPS_3.py: 
finetuning last stage of ConvNeXt foundation model with CASIS-MS-ROI using ArcFace and SupMoCo (supervised MoCo) loss functions.

NIPS_4.py: 
finetuning last stage of ConvNeXt foundation model with CASIS-MS-ROI using ArcFace, SupCon (supervised Contrastive), and GRL (cross-entropy) loss functions.

NIPS_5.py: 
Adding LoRA-MoE to the last stage (stage 3) of ConvNeXt foundation model and fine-tune the model on CASIS-MS-ROI using ArcFace, SupCon (supervised Contrastive), and GRL (cross-entropy) loss functions.

NIPS_6.py: 
Adding LoRA-MoE to the last stage (stage 3) of ConvNeXt foundation model and fine-tune the model on CASIS-MS-ROI using ArcFace, SupCon (supervised Contrastive), and GRL (cross-entropy) loss functions. Then we freeze the whole model except the router (gating network) and perform test-time adaptation (TENT). 

NIPS_7.py: 
Adding LoRA-MoE to the last stage (stage 3) of ConvNeXt foundation model and fine-tune the model on CASIS-MS-ROI using ArcFace, SupCon (supervised Contrastive), and GRL (cross-entropy) loss functions. There is a conflict between GRL and MoE. We use the Scout model to make a better cooperation between them, and achieve a better performance and convergence.   

NIPS_8.py: 
Adding LoRA-MoE to the last stage (stage 3) of ConvNeXt foundation model and fine-tune the model on CASIS-MS-ROI using ArcFace, SupCon (supervised Contrastive), and GRL (cross-entropy) loss functions. We want to use FFT (amplitude/phase) transformation to learn a more general representation.  

