from transformers import StoppingCriteria

class StopOnToken(StoppingCriteria):
    def __init__(self, stop_ids): 
        self.stop_ids=set(stop_ids)

    def __call__(self, input_ids, *_): 
        return input_ids[0, -1].item() in self.stop_ids