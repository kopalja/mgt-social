import torch
from transformers import Trainer

class AyaEncoderTrainer(Trainer):

    encoder_only: bool = True
    def compute_loss(self, model, inputs, return_outputs=False):

        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        labels = inputs.pop("labels")
        if self.encoder_only:
            model_to_use = model.get_encoder()
        else:
            model_to_use = model
            inputs["decoder_input_ids"] = inputs["input_ids"]
            
        outputs = model_to_use(**inputs)
        logits = outputs.last_hidden_state
        
        # compute custom loss. select arbitrary logit to compute loss against
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0, 0], labels.to(torch.float32))
        return (loss, outputs) if return_outputs else loss
        
        
    def predict(self, test_dataset):
        data = torch.tensor(test_dataset['input_ids']).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        if isinstance(self.model, torch.nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
            
        if self.encoder_only:
            model_to_use = model.get_encoder()
        else:
            model_to_use = model
        
        outputs = model_to_use(data)
        print(outputs.last_hidden_state[0])
        return outputs.last_hidden_state[:, 0, 0]