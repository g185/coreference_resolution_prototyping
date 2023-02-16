import numpy as np
from collections import Counter

#Mention evaluation performed as scores on set of mentions.
class MentionEvaluator:
    def __init__(self):
        self.tp, self.fp, self.fn = 0, 0, 0

    #takes predicted mentions and gold mentions:
    #a mention is  [[(wstart, wend)]]
    #calculates tp, fp, and fn as set operations
    #returns prf using formula
    def update(self, predicted_mentions, gold_mentions):
        predicted_mentions = set(predicted_mentions)
        gold_mentions = set(gold_mentions)

        self.tp += len(predicted_mentions & gold_mentions)
        self.fp += len(predicted_mentions - gold_mentions)
        self.fn += len(gold_mentions - predicted_mentions)

    def get_f1(self):
        pr = self.get_precision()
        rec = self.get_recall()
        return 2 * pr * rec / (pr + rec) if pr + rec > 0 else 0.0

    def get_recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    def get_precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def end(self):
        self.tp, self.fp, self.fn = 0,0,0
