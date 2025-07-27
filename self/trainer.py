from trl import SFTTrainer, SFTConfig
import torch
from loguru import logger

class CustomTrainerStage1(SFTTrainer):
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        predict_embeddings = outputs.hidden_states[-1]  # Get last layer hidden states
        image_out_mask = inputs["image_out_mask"] # mask for the latent tokens (B, S)

        # Get the output embeddings of the image tokens (through masking). We exclude the last token
        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device) # latent mask, excluded last token
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous() # output hidden states, excluded last token # (B, S-1, D)

        # Get input embeddings of the image tokens (through masking). We exclude the first token
        input_embeddings = outputs.inputs_embeds
        gt_embeddings = input_embeddings[..., 1:, :][shift_image_mask.to(input_embeddings.device) != 0].contiguous()

        # [0, 1, 2, 3, 4, 5, 6] -> [1, 2, 3, 4, 5]
        # [0, 1, 2, 3, 4, 5, 6] -> [1, 2, 3, 4, 5]
        # Apply the mask here

        sim_loss = torch.nn.functional.cosine_similarity(gt_embeddings, shift_predict_embeddings).mean()
        sim_loss = 1 - sim_loss

        # Weighted loss with 0.1 for the cross entropy loss and 1 for the image embedding loss
        loss = 0.1 * ce_loss + sim_loss
        return (loss, outputs) if return_outputs else loss

class CustomTrainerStage2(SFTTrainer):
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        loss = ce_loss
        return (loss, outputs) if return_outputs else loss